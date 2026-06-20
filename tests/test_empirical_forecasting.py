from tae.connectors.fallback import sample_price_history
from tae.forecasting.empirical import (
    current_bucket_return_distribution,
    empirical_fallback_message,
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
