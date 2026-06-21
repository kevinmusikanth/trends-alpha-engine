import pandas as pd

from tae.connectors.fallback import sample_price_history
from tae.forecasting.empirical import (
    current_bucket_return_distribution,
    empirical_fallback_message,
    empirical_investment_outcome_table,
    empirical_outlook_interpretation,
    empirical_score_bucket_forecast,
    score_bucket_comparison,
)
from tae.forecasting.point_in_time import point_in_time_prediction_frame


def empirical_records():
    prices = {
        ticker: sample_price_history(start="2013-01-01", end="2026-01-01")
        for ticker in ["AAPL", "MSFT", "META"]
    }
    return point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )


def test_empirical_forecast_generation_matches_score_bucket():
    records = empirical_records()
    forecast = empirical_score_bucket_forecast(
        50,
        records,
        investment_amount=10,
        min_observations=1,
    )

    assert not forecast.empty
    assert forecast["score_bucket"].eq("40-60").all()
    assert {
        "average_return_pct",
        "win_rate_pct",
        "observation_count",
        "confidence",
    }.issubset(forecast.columns)


def test_empirical_investment_outcome_conversion():
    records = empirical_records()
    forecast = empirical_score_bucket_forecast(
        50,
        records,
        investment_amount=10,
        min_observations=1,
    )

    one_month = forecast[forecast["horizon"] == "1 month"].iloc[0]
    expected_value = 10 * (1 + one_month["average_return_pct"] / 100)

    assert one_month["expected_value"] == expected_value
    assert one_month["expected_value"] > 0


def test_empirical_forecast_falls_back_when_observations_are_insufficient():
    records = empirical_records()
    forecast = empirical_score_bucket_forecast(
        85,
        records,
        investment_amount=10,
        min_observations=500,
    )

    assert forecast["preferred_forecast"].eq(False).all()
    assert empirical_fallback_message(forecast) == (
        "Insufficient historical observations. Using theoretical forecast."
    )


def test_empirical_charts_have_source_data():
    records = empirical_records()
    comparison = score_bucket_comparison(records)
    distribution = current_bucket_return_distribution(50, records, horizon="12 months")

    assert not comparison.empty
    assert {"score_bucket", "horizon", "average_return_pct"}.issubset(comparison.columns)
    assert not distribution.empty
    assert "actual_return_pct" in distribution.columns


def test_empirical_outlook_interprets_strong_long_term_edge():
    forecast = pd.DataFrame(
        [
            {
                "horizon": "1 month",
                "score_bucket": "60-80",
                "observation_count": 447,
                "average_return_pct": 2.1,
                "median_return_pct": 1.4,
                "win_rate_pct": 57.8,
                "expected_value": 10210,
                "confidence": "Medium",
                "preferred_forecast": True,
                "forecast_error_pct": 10,
                "calibration_accuracy_pct": 90,
            },
            {
                "horizon": "3 months",
                "score_bucket": "60-80",
                "observation_count": 447,
                "average_return_pct": 6.6,
                "median_return_pct": 4.1,
                "win_rate_pct": 60.4,
                "expected_value": 10660,
                "confidence": "Medium",
                "preferred_forecast": True,
                "forecast_error_pct": 10,
                "calibration_accuracy_pct": 90,
            },
            {
                "horizon": "12 months",
                "score_bucket": "60-80",
                "observation_count": 447,
                "average_return_pct": 34.1,
                "median_return_pct": 21.0,
                "win_rate_pct": 76.3,
                "expected_value": 13410,
                "confidence": "High",
                "preferred_forecast": True,
                "forecast_error_pct": 10,
                "calibration_accuracy_pct": 90,
            },
            {
                "horizon": "3 years",
                "score_bucket": "60-80",
                "observation_count": 447,
                "average_return_pct": 134.0,
                "median_return_pct": 88.0,
                "win_rate_pct": 94.7,
                "expected_value": 23400,
                "confidence": "High",
                "preferred_forecast": True,
                "forecast_error_pct": 10,
                "calibration_accuracy_pct": 90,
            },
            {
                "horizon": "5 years",
                "score_bucket": "60-80",
                "observation_count": 447,
                "average_return_pct": 362.1,
                "median_return_pct": 240.0,
                "win_rate_pct": 99.0,
                "expected_value": 46210,
                "confidence": "High",
                "preferred_forecast": True,
                "forecast_error_pct": 10,
                "calibration_accuracy_pct": 90,
            },
        ]
    )

    outlook = empirical_outlook_interpretation(forecast)

    assert outlook["headline"] == (
        "Empirical Outlook: Strong long-term historical edge, "
        "moderate short-term edge."
    )
    assert outlook["confidence"] == "High"
    assert "1 month: 57.8% win rate, +2.1% average return" in outlook["evidence"]


def test_empirical_investment_outcome_table_uses_core_horizons():
    records = empirical_records()
    forecast = empirical_score_bucket_forecast(
        50,
        records,
        investment_amount=10000,
        min_observations=1,
    )

    table = empirical_investment_outcome_table(forecast)

    assert table["horizon"].tolist() == ["1 month", "3 months", "12 months", "3 years", "5 years"]
    assert {"expected_value", "observation_count", "win_rate_pct"}.issubset(table.columns)
    assert table["expected_value"].gt(0).all()


def test_empirical_outlook_flags_insufficient_observations_as_low_confidence():
    records = empirical_records()
    forecast = empirical_score_bucket_forecast(
        85,
        records,
        investment_amount=10,
        min_observations=500,
    )

    outlook = empirical_outlook_interpretation(forecast)

    assert outlook["classification"] == ["High-risk / low-confidence setup"]
    assert outlook["confidence"] == "Low"
