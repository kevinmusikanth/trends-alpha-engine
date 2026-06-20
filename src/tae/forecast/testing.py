from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecast.engine import FORECAST_HORIZONS, forecast_from_score
from tae.scoring.engine import score_ticker


def actual_forward_return(
    price_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    trading_days: int,
) -> float | None:
    prices = price_history.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(prices["date"])
    candidates = prices.index[dates >= as_of_date]
    if len(candidates) == 0:
        return None
    start_index = int(candidates[0])
    end_index = start_index + trading_days
    if end_index >= len(prices):
        return None
    start_price = float(prices.loc[start_index, "close"])
    end_price = float(prices.loc[end_index, "close"])
    if start_price == 0:
        return None
    return end_price / start_price - 1


def prediction_test_frame(
    ticker: str,
    price_history: pd.DataFrame,
    start_date: str,
    step_days: int = 21,
    horizon: str = "3 months",
    fallback_data_used: bool = False,
) -> pd.DataFrame:
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}")

    prices = price_history.sort_values("date").reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"])
    start = pd.Timestamp(start_date)
    rows = []
    trading_days = FORECAST_HORIZONS[horizon]
    eligible = prices.index[prices["date"] >= start]

    for index in eligible[::step_days]:
        if index < 63:
            continue
        as_of_date = prices.loc[index, "date"]
        history = prices.iloc[: index + 1].copy()
        score = score_ticker(
            ticker,
            history,
            live_price_data_available=not fallback_data_used,
            fallback_data_used=fallback_data_used,
        )
        forecast = forecast_from_score(score, history)
        forecast_row = next(row for row in forecast.rows if row["horizon"] == horizon)
        actual = actual_forward_return(prices, as_of_date, trading_days)
        if actual is None:
            continue
        predicted = float(forecast_row["base_case_return"]) / 100
        rows.append(
            {
                "date": as_of_date,
                "forecast_return": predicted,
                "actual_return": actual,
                "error": actual - predicted,
                "confidence_score": forecast.confidence_score,
                "direction_hit": (predicted >= 0 and actual >= 0)
                or (predicted < 0 and actual < 0),
            }
        )
    return pd.DataFrame(rows)


def prediction_test_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "hit_rate": 0.0,
            "average_error": 0.0,
            "sharpe_ratio": 0.0,
            "cagr": 0.0,
            "maximum_drawdown": 0.0,
        }
    returns = frame["actual_return"].astype(float)
    years = max(len(returns) / 12, 1 / 12)
    cumulative = float((1 + returns).prod())
    return {
        "hit_rate": win_rate(frame["direction_hit"].astype(float) - 0.5),
        "average_error": float(frame["error"].abs().mean()),
        "sharpe_ratio": sharpe_ratio(returns, annualization=12),
        "cagr": cumulative ** (1 / years) - 1,
        "maximum_drawdown": max_drawdown(returns),
    }

