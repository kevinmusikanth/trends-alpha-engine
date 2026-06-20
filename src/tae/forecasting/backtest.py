from __future__ import annotations

import pandas as pd

from tae.backtesting.engine import score_band
from tae.backtesting.metrics import max_drawdown, sharpe_ratio
from tae.forecasting.engine import build_forecast_report
from tae.forecasting.models import TRADING_DAYS
from tae.scoring.engine import score_ticker


def actual_forward_return(
    price_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: str,
) -> float | None:
    prices = price_history.sort_values("date").reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"])
    matches = prices.index[prices["date"] >= pd.Timestamp(as_of_date)]
    if len(matches) == 0:
        return None
    start_index = int(matches[0])
    end_index = start_index + TRADING_DAYS[horizon]
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
    horizon: str = "3 months",
    step_days: int = 21,
    fallback_data_used: bool = False,
) -> pd.DataFrame:
    if horizon not in TRADING_DAYS:
        raise ValueError(f"Unsupported prediction test horizon: {horizon}")

    prices = price_history.sort_values("date").reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices["date"])
    rows = []
    eligible = prices.index[prices["date"] >= pd.Timestamp(start_date)]

    for index in eligible[::step_days]:
        if index < 63:
            continue
        as_of_date = prices.loc[index, "date"]
        historical_prices = prices.iloc[: index + 1].copy()
        score = score_ticker(
            ticker,
            historical_prices,
            live_price_data_available=not fallback_data_used,
            fallback_data_used=fallback_data_used,
        )
        report = build_forecast_report(score, historical_prices)
        forecast = next(line for line in report.forecasts if line.horizon == horizon)
        actual = actual_forward_return(prices, as_of_date, horizon)
        if actual is None:
            continue
        predicted = forecast.base_case_pct / 100
        hit = (predicted >= 0 and actual >= 0) or (predicted < 0 and actual < 0)
        rows.append(
            {
                "date": as_of_date,
                "score": score.overall_score,
                "score_bucket": score_band(score.overall_score),
                "predicted_return": predicted,
                "actual_return": actual,
                "error_pct": (actual - predicted) * 100,
                "hit": hit,
                "confidence_pct": forecast.confidence_pct,
                "confidence_bucket": confidence_bucket(forecast.confidence_pct),
            }
        )
    return pd.DataFrame(rows)


def prediction_test_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "average_error": 0.0,
            "hit_rate": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "maximum_drawdown": 0.0,
        }
    returns = frame["actual_return"].astype(float)
    years = max(len(returns) / 12, 1 / 12)
    cumulative = float((1 + returns).prod())
    return {
        "average_error": float(frame["error_pct"].abs().mean()),
        "hit_rate": float(frame["hit"].mean()),
        "cagr": cumulative ** (1 / years) - 1,
        "sharpe_ratio": sharpe_ratio(returns, annualization=12),
        "maximum_drawdown": max_drawdown(returns),
    }


def confidence_bucket(confidence: float) -> str:
    if confidence >= 80:
        return "80 to 100"
    if confidence >= 60:
        return "60 to 79"
    if confidence >= 40:
        return "40 to 59"
    return "Below 40"

