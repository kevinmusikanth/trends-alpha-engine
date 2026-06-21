from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import win_rate
from tae.forecasting.alpha_validation import ALPHA_HORIZONS
from tae.forecasting.point_in_time import prediction_accuracy_metrics
from tae.forecasting.universe import UNIVERSE_SCORE_BUCKETS, score_bucket_20_point

CORE_OUTCOME_HORIZONS = ["1 month", "3 months", "12 months", "3 years", "5 years"]


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


def empirical_investment_outcome_table(
    empirical_forecast: pd.DataFrame,
    horizons: list[str] | None = None,
) -> pd.DataFrame:
    if empirical_forecast.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "expected_value",
                "observation_count",
                "win_rate_pct",
                "average_return_pct",
                "confidence",
            ]
        )

    selected_horizons = horizons or CORE_OUTCOME_HORIZONS
    table = empirical_forecast[
        empirical_forecast["horizon"].isin(selected_horizons)
    ].copy()
    table["horizon"] = pd.Categorical(
        table["horizon"],
        categories=selected_horizons,
        ordered=True,
    )
    columns = [
        "horizon",
        "expected_value",
        "observation_count",
        "win_rate_pct",
        "average_return_pct",
        "confidence",
    ]
    return table.sort_values("horizon")[columns].reset_index(drop=True)


def empirical_outlook_interpretation(empirical_forecast: pd.DataFrame) -> dict[str, object]:
    if empirical_forecast.empty:
        return empirical_outlook_result(
            "No validated edge",
            "Low",
            "There are no historical observations for this score bucket yet.",
            [],
        )

    preferred = empirical_forecast[empirical_forecast["preferred_forecast"]].copy()
    if preferred.empty:
        return empirical_outlook_result(
            "High-risk / low-confidence setup",
            "Low",
            "There are not enough historical observations to validate this score bucket.",
            [],
        )

    evidence = empirical_evidence_lines(preferred)
    long_term = preferred[preferred["horizon"].isin(["12 months", "3 years", "5 years"])]
    short_term = preferred[preferred["horizon"].isin(["1 month", "3 months"])]

    long_term_label = classify_long_term_edge(long_term)
    short_term_label = classify_short_term_edge(short_term)
    confidence = empirical_overall_confidence(preferred)

    labels = [label for label in [long_term_label, short_term_label] if label]
    if not labels:
        labels = ["No validated edge"]

    if confidence == "Low" and "No validated edge" not in labels:
        labels.append("High-risk / low-confidence setup")

    headline = "Empirical Outlook: " + ", ".join(
        format_outlook_label(label, index) for index, label in enumerate(labels)
    ) + "."
    explanation = build_empirical_explanation(labels, confidence)

    return {
        "headline": headline,
        "classification": labels,
        "confidence": confidence,
        "explanation": explanation,
        "evidence": evidence,
    }


def empirical_outlook_result(
    label: str,
    confidence: str,
    explanation: str,
    evidence: list[str],
) -> dict[str, object]:
    return {
        "headline": f"Empirical Outlook: {label}.",
        "classification": [label],
        "confidence": confidence,
        "explanation": explanation,
        "evidence": evidence,
    }


def empirical_evidence_lines(empirical_forecast: pd.DataFrame) -> list[str]:
    lines = []
    for horizon in CORE_OUTCOME_HORIZONS:
        row = empirical_forecast[empirical_forecast["horizon"] == horizon]
        if row.empty:
            continue
        item = row.iloc[0]
        lines.append(
            f"{horizon}: {item['win_rate_pct']:.1f}% win rate, "
            f"{item['average_return_pct']:+.1f}% average return"
        )
    return lines


def classify_long_term_edge(long_term: pd.DataFrame) -> str | None:
    if long_term.empty:
        return None
    strong = long_term[
        (long_term["average_return_pct"] >= 15)
        & (long_term["win_rate_pct"] >= 65)
    ]
    moderate = long_term[
        (long_term["average_return_pct"] >= 5)
        & (long_term["win_rate_pct"] >= 55)
    ]
    if len(strong) >= 2 or (
        not strong.empty
        and strong["average_return_pct"].max() >= 25
        and strong["win_rate_pct"].max() >= 70
    ):
        return "Strong long-term historical edge"
    if len(moderate) >= 1:
        return "Moderate long-term edge"
    return None


def classify_short_term_edge(short_term: pd.DataFrame) -> str | None:
    if short_term.empty:
        return None
    moderate = short_term[
        (short_term["average_return_pct"] >= 1)
        & (short_term["win_rate_pct"] >= 55)
    ]
    weak = short_term[
        (short_term["average_return_pct"] > 0)
        & (short_term["win_rate_pct"] >= 50)
    ]
    if not moderate.empty:
        return "Moderate short-term edge"
    if not weak.empty:
        return "Weak short-term edge"
    return None


def empirical_overall_confidence(empirical_forecast: pd.DataFrame) -> str:
    if empirical_forecast.empty:
        return "Low"
    confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
    median_observations = float(empirical_forecast["observation_count"].median())
    average_win_rate = float(empirical_forecast["win_rate_pct"].mean())
    average_calibration = float(empirical_forecast["calibration_accuracy_pct"].mean())
    best_confidence = int(
        empirical_forecast["confidence"].map(confidence_rank).fillna(0).max()
    )

    if (
        best_confidence >= confidence_rank["High"]
        or (
            median_observations >= 100
            and average_win_rate >= 60
            and average_calibration >= 70
        )
    ):
        return "High"
    if (
        best_confidence >= confidence_rank["Medium"]
        or (
            median_observations >= 30
            and average_win_rate >= 52
            and average_calibration >= 55
        )
    ):
        return "Medium"
    return "Low"


def format_outlook_label(label: str, index: int) -> str:
    if index == 0:
        return label
    return label[:1].lower() + label[1:]


def build_empirical_explanation(labels: list[str], confidence: str) -> str:
    if "No validated edge" in labels:
        return (
            "Historical outcomes for this score bucket do not yet show a reliable "
            "positive edge."
        )
    if "High-risk / low-confidence setup" in labels:
        return (
            "The bucket has some positive evidence, but the sample quality or "
            "calibration is not strong enough to treat it as a high-confidence setup."
        )
    return (
        "The empirical outlook is based on realized point-in-time outcomes for "
        "stocks in the same score bucket. Confidence reflects observation count, "
        f"win rate, consistency and calibration quality. Current confidence: {confidence}."
    )


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
