from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import win_rate
from tae.forecasting.alpha_validation import ALPHA_HORIZONS
from tae.forecasting.point_in_time import prediction_accuracy_metrics
from tae.forecasting.universe import UNIVERSE_SCORE_BUCKETS, score_bucket_20_point


def empirical_score_bucket_forecast(
    score: float,
    validation_records: pd.DataFrame,
    investment_amount: float = 10.0,
    min_observations: int = 20,
) -> pd.DataFrame:
    bucket = score_bucket_20_point(score)
    rows = []
    if validation_records.empty:
        return empty_empirical_forecast(bucket)

    bucket_records = validation_records[validation_records["score_bucket"] == bucket]
    overall_accuracy = prediction_accuracy_metrics(validation_records)

    for horizon in ALPHA_HORIZONS:
        group = bucket_records[bucket_records["horizon"] == horizon]
        if group.empty:
            rows.append(empty_horizon_row(bucket, horizon, min_observations))
            continue

        returns = group["actual_future_return"].astype(float)
        average_return = float(returns.mean())
        median_return = float(returns.median())
        win_rate_pct = float(win_rate(returns) * 100)
        observation_count = int(len(group))
        forecast_error_pct = empirical_forecast_error_pct(group)
        calibration_accuracy_pct = max(0.0, 100 - forecast_error_pct)
        confidence = empirical_confidence_level(
            observation_count,
            win_rate_pct,
            consistency_pct=float(max(0.0, 100 - returns.std() * 100)),
            forecast_error_pct=forecast_error_pct,
            calibration_accuracy_pct=calibration_accuracy_pct,
            min_observations=min_observations,
        )
        rows.append(
            {
                "horizon": horizon,
                "score_bucket": bucket,
                "observation_count": observation_count,
                "average_return_pct": average_return * 100,
                "median_return_pct": median_return * 100,
                "win_rate_pct": win_rate_pct,
                "expected_value": investment_amount * (1 + average_return),
                "confidence": confidence,
                "preferred_forecast": observation_count >= min_observations,
                "forecast_error_pct": forecast_error_pct,
                "calibration_accuracy_pct": calibration_accuracy_pct,
                "overall_calibration_accuracy_pct": overall_accuracy[
                    "calibration_accuracy_pct"
                ],
            }
        )

    return pd.DataFrame(rows)


def empty_empirical_forecast(bucket: str) -> pd.DataFrame:
    return pd.DataFrame(
        [empty_horizon_row(bucket, horizon, min_observations=0) for horizon in ALPHA_HORIZONS]
    )


def empty_horizon_row(bucket: str, horizon: str, min_observations: int) -> dict[str, object]:
    return {
        "horizon": horizon,
        "score_bucket": bucket,
        "observation_count": 0,
        "average_return_pct": 0.0,
        "median_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "expected_value": 0.0,
        "confidence": "Low",
        "preferred_forecast": False,
        "forecast_error_pct": 100.0 if min_observations else 0.0,
        "calibration_accuracy_pct": 0.0,
        "overall_calibration_accuracy_pct": 0.0,
    }


def empirical_forecast_error_pct(group: pd.DataFrame) -> float:
    if "prediction_error" not in group or group.empty:
        return 100.0
    return float(group["prediction_error"].abs().mean() * 100)


def empirical_confidence_level(
    observation_count: int,
    win_rate_pct: float,
    consistency_pct: float,
    forecast_error_pct: float,
    calibration_accuracy_pct: float,
    min_observations: int = 20,
) -> str:
    if observation_count < min_observations:
        return "Low"
    if (
        observation_count >= min_observations * 3
        and win_rate_pct >= 60
        and consistency_pct >= 70
        and forecast_error_pct <= 12
        and calibration_accuracy_pct >= 80
    ):
        return "High"
    if win_rate_pct >= 50 and consistency_pct >= 55 and calibration_accuracy_pct >= 65:
        return "Medium"
    return "Low"


def empirical_fallback_message(empirical_forecast: pd.DataFrame) -> str:
    if empirical_forecast.empty or not empirical_forecast["preferred_forecast"].any():
        return "Insufficient historical observations. Using theoretical forecast."
    return "Empirical score-bucket forecast is preferred for horizons with enough observations."


def score_bucket_comparison(validation_records: pd.DataFrame) -> pd.DataFrame:
    if validation_records.empty:
        return pd.DataFrame()
    rows = []
    for (bucket, horizon), group in validation_records.groupby(
        ["score_bucket", "horizon"],
        sort=False,
    ):
        returns = group["actual_future_return"].astype(float)
        rows.append(
            {
                "score_bucket": bucket,
                "horizon": horizon,
                "observation_count": int(len(group)),
                "average_return_pct": float(returns.mean() * 100),
                "win_rate_pct": float(win_rate(returns) * 100),
            }
        )
    frame = pd.DataFrame(rows)
    frame["score_bucket"] = pd.Categorical(
        frame["score_bucket"],
        categories=UNIVERSE_SCORE_BUCKETS,
        ordered=True,
    )
    frame["horizon"] = pd.Categorical(frame["horizon"], categories=ALPHA_HORIZONS, ordered=True)
    return frame.sort_values(["horizon", "score_bucket"]).reset_index(drop=True)


def current_bucket_return_distribution(
    score: float,
    validation_records: pd.DataFrame,
    horizon: str = "12 months",
) -> pd.DataFrame:
    if validation_records.empty:
        return pd.DataFrame(columns=["actual_return_pct"])
    bucket = score_bucket_20_point(score)
    group = validation_records[
        (validation_records["score_bucket"] == bucket)
        & (validation_records["horizon"] == horizon)
    ]
    return pd.DataFrame({"actual_return_pct": group["actual_future_return"] * 100})
