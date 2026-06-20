from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecasting.backtest import prediction_test_frame

UNIVERSE_HORIZONS = ["1 month", "3 months", "6 months", "12 months"]
UNIVERSE_SCORE_BUCKETS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def score_bucket_20_point(score: float) -> str:
    score = max(0.0, min(100.0, float(score)))
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def universe_prediction_frame(
    price_history_by_ticker: dict[str, pd.DataFrame],
    start_date: str,
    fallback_data_used_by_ticker: dict[str, bool] | None = None,
    horizons: list[str] | None = None,
    step_days: int = 21,
) -> pd.DataFrame:
    horizons = horizons or UNIVERSE_HORIZONS
    fallback_data_used_by_ticker = fallback_data_used_by_ticker or {}
    frames = []

    for ticker, prices in price_history_by_ticker.items():
        is_fallback = bool(fallback_data_used_by_ticker.get(ticker, False))
        for horizon in horizons:
            frame = prediction_test_frame(
                ticker,
                prices,
                start_date=start_date,
                horizon=horizon,
                step_days=step_days,
                fallback_data_used=is_fallback,
            )
            if frame.empty:
                continue
            frame = frame.copy()
            frame["ticker"] = ticker
            frame["horizon"] = horizon
            frame["score_bucket"] = frame["score"].map(score_bucket_20_point)
            frame["fallback_data_used"] = is_fallback
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "horizon",
                "score",
                "score_bucket",
                "predicted_return",
                "actual_return",
                "error_pct",
                "hit",
                "confidence_pct",
                "fallback_data_used",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def universe_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "score_bucket",
                "sample_size",
                "average_return_pct",
                "median_return_pct",
                "win_rate_pct",
                "sharpe_ratio",
                "maximum_drawdown_pct",
            ]
        )

    rows = []
    for (horizon, bucket), group in frame.groupby(["horizon", "score_bucket"], sort=False):
        returns = group["actual_return"].astype(float)
        rows.append(
            {
                "horizon": horizon,
                "score_bucket": bucket,
                "sample_size": int(len(group)),
                "average_return_pct": float(returns.mean() * 100),
                "median_return_pct": float(returns.median() * 100),
                "win_rate_pct": float(win_rate(returns) * 100),
                "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                "maximum_drawdown_pct": float(max_drawdown(returns) * 100),
            }
        )

    summary = pd.DataFrame(rows)
    summary["score_bucket"] = pd.Categorical(
        summary["score_bucket"],
        categories=UNIVERSE_SCORE_BUCKETS,
        ordered=True,
    )
    summary["horizon"] = pd.Categorical(
        summary["horizon"],
        categories=UNIVERSE_HORIZONS,
        ordered=True,
    )
    return summary.sort_values(["horizon", "score_bucket"]).reset_index(drop=True)


def universe_calibration_curve(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "score_bucket",
                "sample_size",
                "average_predicted_return_pct",
                "average_actual_return_pct",
                "forecast_bias_pct",
            ]
        )

    rows = []
    for (horizon, bucket), group in frame.groupby(["horizon", "score_bucket"], sort=False):
        predicted = group["predicted_return"].astype(float)
        actual = group["actual_return"].astype(float)
        rows.append(
            {
                "horizon": horizon,
                "score_bucket": bucket,
                "sample_size": int(len(group)),
                "average_predicted_return_pct": float(predicted.mean() * 100),
                "average_actual_return_pct": float(actual.mean() * 100),
                "forecast_bias_pct": float((predicted.mean() - actual.mean()) * 100),
            }
        )

    calibration = pd.DataFrame(rows)
    calibration["score_bucket"] = pd.Categorical(
        calibration["score_bucket"],
        categories=UNIVERSE_SCORE_BUCKETS,
        ordered=True,
    )
    calibration["horizon"] = pd.Categorical(
        calibration["horizon"],
        categories=UNIVERSE_HORIZONS,
        ordered=True,
    )
    return calibration.sort_values(["horizon", "score_bucket"]).reset_index(drop=True)
