from __future__ import annotations

import numpy as np
import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecasting.backtest import prediction_test_frame

SCORE_BUCKETS = [
    (90, 100, "90-100"),
    (80, 89.999, "80-90"),
    (70, 79.999, "70-80"),
    (60, 69.999, "60-70"),
    (-np.inf, 59.999, "below 60"),
]

FORECAST_BUCKETS = [
    (-np.inf, 0.04999, "0-5%"),
    (0.05, 0.09999, "5-10%"),
    (0.10, 0.14999, "10-15%"),
    (0.15, 0.19999, "15-20%"),
    (0.20, np.inf, "20%+"),
]

VALIDATION_HORIZONS = ["1 week", "1 month", "3 months", "6 months", "12 months"]
FACTOR_COLUMNS = [
    "detailed_factor_momentum",
    "detailed_factor_relative_strength",
    "detailed_factor_volume",
    "detailed_factor_revenue_growth",
    "detailed_factor_eps_growth",
    "detailed_factor_margin_expansion",
    "detailed_factor_roic",
    "detailed_factor_free_cash_flow_growth",
    "detailed_factor_institutional_buying",
    "detailed_factor_analyst_revisions",
    "detailed_factor_valuation",
    "detailed_factor_quality",
    "detailed_factor_risk",
]


def validation_frame(
    ticker: str,
    price_history: pd.DataFrame,
    start_date: str,
    fallback_data_used: bool = False,
) -> pd.DataFrame:
    frames = []
    for horizon in VALIDATION_HORIZONS:
        frame = prediction_test_frame(
            ticker,
            price_history,
            start_date=start_date,
            horizon=horizon,
            fallback_data_used=fallback_data_used,
        )
        if not frame.empty:
            frame["horizon"] = horizon
            frame["validation_score_bucket"] = frame["score"].apply(validation_score_bucket)
            frame["forecast_bucket"] = frame["predicted_return"].apply(forecast_bucket)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def validation_score_bucket(score: float) -> str:
    for low, high, label in SCORE_BUCKETS:
        if low <= score <= high:
            return label
    return "below 60"


def forecast_bucket(predicted_return: float) -> str:
    for low, high, label in FORECAST_BUCKETS:
        if low <= predicted_return <= high:
            return label
    return "0-5%"


def score_bucket_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(["horizon", "validation_score_bucket"], sort=False)
    for (horizon, bucket), group in grouped:
        returns = group["actual_return"].astype(float)
        rows.append(
            {
                "horizon": horizon,
                "score_bucket": bucket,
                "average_forward_return_pct": returns.mean() * 100,
                "median_return_pct": returns.median() * 100,
                "win_rate_pct": win_rate(returns) * 100,
                "volatility_pct": returns.std() * 100,
                "max_drawdown_pct": max_drawdown(returns) * 100,
                "sample_size": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def forecast_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    grouped = frame.groupby(["horizon", "forecast_bucket"], sort=False)
    for (horizon, bucket), group in grouped:
        rows.append(
            {
                "horizon": horizon,
                "forecast_bucket": bucket,
                "average_predicted_return_pct": group["predicted_return"].mean() * 100,
                "average_actual_return_pct": group["actual_return"].mean() * 100,
                "forecast_bias_pct": (
                    group["predicted_return"].mean() - group["actual_return"].mean()
                )
                * 100,
                "hit_rate_pct": group["hit"].mean() * 100,
                "sample_size": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def confidence_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["horizon", "confidence_bucket"], sort=False)
    return grouped.agg(
        average_confidence_pct=("confidence_pct", "mean"),
        actual_hit_rate_pct=("hit", lambda series: series.mean() * 100),
        sample_size=("hit", "size"),
    ).reset_index()


def feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for factor in FACTOR_COLUMNS:
        if factor not in frame.columns or frame[factor].std() == 0:
            importance = 0.0
        else:
            importance = frame[factor].corr(frame["actual_return"])
            if np.isnan(importance):
                importance = 0.0
        rows.append(
            {
                "factor": factor.replace("detailed_factor_", "").replace("_", " ").title(),
                "importance": abs(float(importance)),
                "correlation_to_forward_return": float(importance),
                "predictive_power": predictive_power_label(float(importance)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def model_quality_metrics(frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty:
        return {
            "r_squared": 0.0,
            "mean_absolute_error_pct": 0.0,
            "rmse_pct": 0.0,
            "hit_rate_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "information_ratio": 0.0,
            "maximum_drawdown_pct": 0.0,
            "calibration_error_pct": 0.0,
            "predictive_power": "Insufficient data",
            "recalibration_flag": "Model requires recalibration.",
            "recalibration_reasons": "Insufficient validation data",
        }
    predicted = frame["predicted_return"].astype(float)
    actual = frame["actual_return"].astype(float)
    errors = actual - predicted
    ss_res = float((errors**2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r_squared = 0.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
    hit_rate = float(frame["hit"].mean())
    sharpe = sharpe_ratio(actual, annualization=12)
    sortino = sortino_ratio(actual, annualization=12)
    information = information_ratio(actual, predicted, annualization=12)
    drawdown = max_drawdown(actual) * 100
    calibration_error_value = calibration_error(frame)
    power = predictive_power_assessment(r_squared, hit_rate, calibration_error_value)
    reasons = recalibration_reasons(r_squared, hit_rate, calibration_error_value, frame)
    requires_recalibration = bool(reasons)
    return {
        "r_squared": r_squared,
        "mean_absolute_error_pct": float(errors.abs().mean() * 100),
        "rmse_pct": float(np.sqrt((errors**2).mean()) * 100),
        "hit_rate_pct": hit_rate * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "information_ratio": information,
        "maximum_drawdown_pct": drawdown,
        "calibration_error_pct": calibration_error_value,
        "predictive_power": power,
        "recalibration_flag": (
            "Model requires recalibration." if requires_recalibration else ""
        ),
        "recalibration_reasons": "; ".join(reasons),
    }


def sortino_ratio(returns: pd.Series, annualization: float = 252) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    downside_std = downside.std()
    if returns.empty or downside.empty or downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float((returns.mean() / downside_std) * np.sqrt(annualization))


def information_ratio(
    actual_returns: pd.Series,
    predicted_returns: pd.Series,
    annualization: float = 252,
) -> float:
    active_return = actual_returns.dropna() - predicted_returns.dropna()
    tracking_error = active_return.std()
    if active_return.empty or tracking_error == 0 or np.isnan(tracking_error):
        return 0.0
    return float((active_return.mean() / tracking_error) * np.sqrt(annualization))


def calibration_error(frame: pd.DataFrame) -> float:
    calibration = forecast_calibration(frame)
    if calibration.empty:
        return 0.0
    errors = (
        calibration["average_actual_return_pct"]
        - calibration["average_predicted_return_pct"]
    ).abs()
    weights = calibration["sample_size"] / calibration["sample_size"].sum()
    return float((errors * weights).sum())


def predictive_power_assessment(
    r_squared: float,
    hit_rate: float,
    calibration_error_pct: float,
) -> str:
    if r_squared >= 0.15 and hit_rate >= 0.57 and calibration_error_pct <= 7.5:
        return "Strong Predictive Power"
    if r_squared >= 0.05 and hit_rate >= 0.52 and calibration_error_pct <= 12.0:
        return "Moderate Predictive Power"
    return "Weak Predictive Power"


def recalibration_reasons(
    r_squared: float,
    hit_rate: float,
    calibration_error_pct: float,
    frame: pd.DataFrame,
) -> list[str]:
    reasons = []
    if r_squared < 0.05:
        reasons.append("Weak predictive power")
    if hit_rate < 0.52:
        reasons.append("Low hit rate")
    if calibration_error_pct > 12.0:
        reasons.append("High forecast error")
    confidence = confidence_calibration(frame)
    if not confidence.empty:
        confidence_gap = (
            confidence["average_confidence_pct"] - confidence["actual_hit_rate_pct"]
        ).abs().mean()
        if confidence_gap > 25:
            reasons.append("Poor confidence calibration")
    importance = feature_importance(frame)
    if not importance.empty and importance["importance"].head(3).mean() < 0.05:
        reasons.append("Degraded factor effectiveness")
    return reasons


def predictive_power_label(correlation: float) -> str:
    absolute = abs(correlation)
    if absolute >= 0.20:
        return "Strong"
    if absolute >= 0.08:
        return "Moderate"
    return "Weak"
