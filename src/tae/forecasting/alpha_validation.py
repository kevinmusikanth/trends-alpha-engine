from __future__ import annotations

import numpy as np
import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecasting.backtest import normalize_price_dates, normalize_timestamp
from tae.forecasting.engine import build_forecast_report
from tae.forecasting.universe import UNIVERSE_SCORE_BUCKETS, score_bucket_20_point
from tae.scoring.engine import score_ticker

ALPHA_HORIZONS = ["1 week", "1 month", "3 months", "6 months", "12 months", "3 years", "5 years"]
ALPHA_HORIZON_DAYS = {
    "1 week": 5,
    "1 month": 21,
    "3 months": 63,
    "6 months": 126,
    "12 months": 252,
    "3 years": 756,
    "5 years": 1260,
}
FORECAST_HORIZON_MAP = {
    "1 week": ("1 week", 0.019, False),
    "1 month": ("1 month", 1 / 12, False),
    "3 months": ("3 months", 0.25, False),
    "6 months": ("6 months", 0.5, False),
    "12 months": ("12 months", 1.0, False),
    "3 years": ("3 year CAGR", 3.0, True),
    "5 years": ("5 year CAGR", 5.0, True),
}


def alpha_validation_frame(
    price_history_by_ticker: dict[str, pd.DataFrame],
    start_date: str,
    fallback_data_used_by_ticker: dict[str, bool] | None = None,
    horizons: list[str] | None = None,
    step_days: int = 21,
) -> pd.DataFrame:
    horizons = horizons or ALPHA_HORIZONS
    fallback_data_used_by_ticker = fallback_data_used_by_ticker or {}
    rows = []

    for ticker, price_history in price_history_by_ticker.items():
        prices = normalize_price_dates(price_history)
        if prices.empty:
            continue
        start = normalize_timestamp(start_date)
        eligible = prices.index[prices["date"] >= start]
        is_fallback = bool(fallback_data_used_by_ticker.get(ticker, False))

        for index in eligible[:: max(1, step_days)]:
            if index < 63:
                continue
            historical_prices = prices.iloc[: index + 1].copy()
            score = score_ticker(
                ticker,
                historical_prices,
                live_price_data_available=not is_fallback,
                fallback_data_used=is_fallback,
            )
            report = build_forecast_report(score, historical_prices)
            as_of_date = prices.loc[index, "date"]

            for horizon in horizons:
                actual = actual_forward_return_by_days(prices, int(index), horizon)
                if actual is None:
                    continue
                predicted = predicted_return_for_horizon(report, horizon)
                drawdown, volatility = forward_window_risk(prices, int(index), horizon)
                rows.append(
                    {
                        "ticker": ticker,
                        "date": as_of_date,
                        "horizon": horizon,
                        "score": score.overall_score,
                        "score_bucket": score_bucket_20_point(score.overall_score),
                        "predicted_return": predicted,
                        "actual_return": actual,
                        "win_loss": "Win" if actual > 0 else "Loss",
                        "hit": bool(actual > 0),
                        "drawdown": drawdown,
                        "volatility": volatility,
                        "fallback_data_used": is_fallback,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "horizon",
                "score",
                "score_bucket",
                "predicted_return",
                "actual_return",
                "win_loss",
                "hit",
                "drawdown",
                "volatility",
                "fallback_data_used",
            ]
        )
    return pd.DataFrame(rows)


def predicted_return_for_horizon(report, horizon: str) -> float:
    forecast_horizon, years, is_cagr = FORECAST_HORIZON_MAP[horizon]
    forecast = next(line for line in report.forecasts if line.horizon == forecast_horizon)
    rate = forecast.base_case_pct / 100
    if is_cagr:
        return float((1 + rate) ** years - 1)
    return float(rate)


def actual_forward_return_by_days(
    prices: pd.DataFrame,
    start_index: int,
    horizon: str,
) -> float | None:
    end_index = start_index + ALPHA_HORIZON_DAYS[horizon]
    if end_index >= len(prices):
        return None
    start_price = float(prices.loc[start_index, "close"])
    end_price = float(prices.loc[end_index, "close"])
    if start_price == 0:
        return None
    return end_price / start_price - 1


def forward_window_risk(prices: pd.DataFrame, start_index: int, horizon: str) -> tuple[float, float]:
    end_index = start_index + ALPHA_HORIZON_DAYS[horizon]
    if end_index >= len(prices):
        return 0.0, 0.0
    window = prices.iloc[start_index : end_index + 1]["close"].astype(float)
    returns = window.pct_change().dropna()
    if returns.empty:
        return 0.0, 0.0
    return max_drawdown(returns), float(returns.std() * np.sqrt(252))


def score_bucket_performance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_bucket_performance()

    rows = []
    for (horizon, bucket), group in frame.groupby(["horizon", "score_bucket"], sort=False):
        returns = group["actual_return"].astype(float)
        rows.append(
            {
                "horizon": horizon,
                "score_bucket": bucket,
                "observation_count": int(len(group)),
                "average_return_pct": float(returns.mean() * 100),
                "median_return_pct": float(returns.median() * 100),
                "win_rate_pct": float(win_rate(returns) * 100),
                "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                "maximum_drawdown_pct": float(group["drawdown"].min() * 100),
                "volatility_pct": float(group["volatility"].mean() * 100),
            }
        )
    return sort_bucket_table(pd.DataFrame(rows))


def empty_bucket_performance() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "score_bucket",
            "observation_count",
            "average_return_pct",
            "median_return_pct",
            "win_rate_pct",
            "sharpe_ratio",
            "maximum_drawdown_pct",
            "volatility_pct",
        ]
    )


def sort_bucket_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["score_bucket"] = pd.Categorical(
        frame["score_bucket"],
        categories=UNIVERSE_SCORE_BUCKETS,
        ordered=True,
    )
    frame["horizon"] = pd.Categorical(frame["horizon"], categories=ALPHA_HORIZONS, ordered=True)
    return frame.sort_values(["horizon", "score_bucket"]).reset_index(drop=True)


def predictor_validation_metrics(frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty:
        return {
            "correlation": 0.0,
            "r_squared": 0.0,
            "information_coefficient": 0.0,
            "hit_rate_pct": 0.0,
            "calibration_accuracy_pct": 0.0,
            "predictive_power": "Insufficient data",
        }

    score = frame["score"].astype(float)
    actual = frame["actual_return"].astype(float)
    predicted = frame["predicted_return"].astype(float)
    correlation = safe_corr(score, actual)
    errors = actual - predicted
    ss_res = float((errors**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r_squared = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    calibration_accuracy = max(0.0, 100 - float(errors.abs().mean() * 100))
    hit_rate = win_rate(actual) * 100
    return {
        "correlation": correlation,
        "r_squared": r_squared,
        "information_coefficient": correlation,
        "hit_rate_pct": hit_rate,
        "calibration_accuracy_pct": calibration_accuracy,
        "predictive_power": predictive_power_label(correlation, r_squared, hit_rate),
    }


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty or left.std() == 0 or right.std() == 0:
        return 0.0
    correlation = left.corr(right)
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def predictive_power_label(correlation: float, r_squared: float, hit_rate_pct: float) -> str:
    if correlation >= 0.20 and r_squared >= 0.08 and hit_rate_pct >= 57:
        return "Strong Predictive Power"
    if correlation >= 0.08 and r_squared >= 0.02 and hit_rate_pct >= 52:
        return "Moderate Predictive Power"
    return "Weak Predictive Power"


def investment_outcome_by_bucket(
    bucket_performance: pd.DataFrame,
    investment_amount: float,
) -> pd.DataFrame:
    if bucket_performance.empty:
        return pd.DataFrame()
    frame = bucket_performance.copy()
    frame["expected_value"] = investment_amount * (1 + frame["average_return_pct"] / 100)
    frame["profit_loss"] = frame["expected_value"] - investment_amount
    return frame[
        [
            "horizon",
            "score_bucket",
            "observation_count",
            "average_return_pct",
            "win_rate_pct",
            "expected_value",
            "profit_loss",
        ]
    ]


def threshold_analysis(frame: pd.DataFrame, thresholds: list[int] | None = None) -> pd.DataFrame:
    thresholds = thresholds or [60, 70, 80, 90]
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for threshold in thresholds:
        eligible = frame[frame["score"] > threshold]
        for horizon, group in eligible.groupby("horizon", sort=False):
            returns = group["actual_return"].astype(float)
            years = horizon_years(horizon)
            rows.append(
                {
                    "threshold": f"Score > {threshold}",
                    "horizon": horizon,
                    "opportunity_count": int(len(group)),
                    "average_return_pct": float(returns.mean() * 100),
                    "win_rate_pct": float(win_rate(returns) * 100),
                    "cagr_pct": annualized_return(returns.mean(), years) * 100,
                    "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                    "maximum_drawdown_pct": float(group["drawdown"].min() * 100),
                }
            )
    return pd.DataFrame(rows)


def annualized_return(total_return: float, years: float) -> float:
    if years <= 0 or total_return <= -1:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def horizon_years(horizon: str) -> float:
    return {
        "1 week": 5 / 252,
        "1 month": 1 / 12,
        "3 months": 0.25,
        "6 months": 0.5,
        "12 months": 1.0,
        "3 years": 3.0,
        "5 years": 5.0,
    }[horizon]


def benchmark_return_frame(
    name: str,
    price_history: pd.DataFrame,
    start_date: str,
    horizons: list[str] | None = None,
    step_days: int = 21,
) -> pd.DataFrame:
    horizons = horizons or ALPHA_HORIZONS
    prices = normalize_price_dates(price_history)
    if prices.empty:
        return pd.DataFrame()
    start = normalize_timestamp(start_date)
    eligible = prices.index[prices["date"] >= start]
    rows = []
    for index in eligible[:: max(1, step_days)]:
        for horizon in horizons:
            actual = actual_forward_return_by_days(prices, int(index), horizon)
            if actual is None:
                continue
            drawdown, volatility = forward_window_risk(prices, int(index), horizon)
            rows.append(
                {
                    "benchmark": name,
                    "date": prices.loc[index, "date"],
                    "horizon": horizon,
                    "actual_return": actual,
                    "drawdown": drawdown,
                    "volatility": volatility,
                }
            )
    return pd.DataFrame(rows)


def benchmark_comparison(
    frame: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    equal_weight = frame.groupby(["date", "horizon"])["actual_return"].mean().reset_index()
    benchmark_specs = {"Equal-weight universe": equal_weight}
    if benchmark_frames:
        benchmark_specs.update(benchmark_frames)
    for name, benchmark in benchmark_specs.items():
        if benchmark.empty:
            continue
        for horizon, group in benchmark.groupby("horizon", sort=False):
            returns = group["actual_return"].astype(float)
            universe_return = equal_weight[equal_weight["horizon"] == horizon][
                "actual_return"
            ].mean()
            benchmark_return = returns.mean()
            years = horizon_years(horizon)
            cagr = annualized_return(benchmark_return, years)
            rows.append(
                {
                    "benchmark": name,
                    "horizon": horizon,
                    "cagr_pct": cagr * 100,
                    "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                    "maximum_drawdown_pct": max_drawdown(returns) * 100,
                    "alpha_pct": (universe_return - benchmark_return) * 100,
                    "excess_return_pct": (universe_return - benchmark_return) * 100,
                }
            )
    return pd.DataFrame(rows)


def confidence_framework(
    bucket_performance: pd.DataFrame,
    score: float,
    horizon: str = "12 months",
) -> dict[str, float | str]:
    if bucket_performance.empty:
        return empty_confidence_framework(score, horizon)
    bucket = score_bucket_20_point(score)
    match = bucket_performance[
        (bucket_performance["score_bucket"] == bucket)
        & (bucket_performance["horizon"] == horizon)
    ]
    if match.empty:
        return empty_confidence_framework(score, horizon, bucket)
    row = match.iloc[0]
    return {
        "score": score,
        "score_bucket": bucket,
        "horizon": horizon,
        "historical_observations": int(row["observation_count"]),
        "probability_positive_pct": float(row["win_rate_pct"]),
        "average_return_pct": float(row["average_return_pct"]),
        "median_return_pct": float(row["median_return_pct"]),
        "probability_loss_pct": float(100 - row["win_rate_pct"]),
        "confidence": confidence_label(int(row["observation_count"]), float(row["win_rate_pct"])),
    }


def empty_confidence_framework(
    score: float,
    horizon: str,
    bucket: str | None = None,
) -> dict[str, float | str]:
    return {
        "score": score,
        "score_bucket": bucket or score_bucket_20_point(score),
        "horizon": horizon,
        "historical_observations": 0,
        "probability_positive_pct": 0.0,
        "average_return_pct": 0.0,
        "median_return_pct": 0.0,
        "probability_loss_pct": 0.0,
        "confidence": "Low",
    }


def confidence_label(observations: int, win_rate_pct: float) -> str:
    if observations >= 100 and win_rate_pct >= 60:
        return "High"
    if observations >= 30 and win_rate_pct >= 50:
        return "Medium"
    return "Low"


def final_investment_outcome_card(
    bucket_performance: pd.DataFrame,
    score: float,
    investment_amount: float,
) -> pd.DataFrame:
    if bucket_performance.empty:
        return pd.DataFrame()
    bucket = score_bucket_20_point(score)
    frame = bucket_performance[bucket_performance["score_bucket"] == bucket].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["expected_value"] = investment_amount * (1 + frame["average_return_pct"] / 100)
    framework = confidence_framework(bucket_performance, score)
    frame["historical_observations_12m"] = framework["historical_observations"]
    frame["historical_win_rate_12m_pct"] = framework["probability_positive_pct"]
    frame["confidence"] = framework["confidence"]
    return frame[
        [
            "horizon",
            "score_bucket",
            "average_return_pct",
            "expected_value",
            "historical_win_rate_12m_pct",
            "historical_observations_12m",
            "confidence",
        ]
    ]
