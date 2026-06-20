from tae.connectors.fallback import sample_price_history
from tae.forecasting.validation import (
    confidence_calibration,
    feature_importance,
    forecast_calibration,
    model_quality_metrics,
    score_bucket_analysis,
    validation_frame,
)


def test_validation_frame_builds_all_required_horizons_with_factors():
    prices = sample_price_history(start="2019-01-01", end="2024-12-31")
    frame = validation_frame(
        "MSFT",
        prices,
        start_date="2020-01-01",
        fallback_data_used=True,
    )

    assert not frame.empty
    assert {"1 week", "1 month", "3 months", "6 months", "12 months"}.issubset(
        set(frame["horizon"])
    )
    assert "factor_momentum" in frame.columns
    assert "detailed_factor_revenue_growth" in frame.columns
    assert "detailed_factor_eps_growth" in frame.columns
    assert "detailed_factor_roic" in frame.columns
    assert "validation_score_bucket" in frame.columns
    assert "forecast_bucket" in frame.columns


def test_score_bucket_and_forecast_calibration_tables():
    prices = sample_price_history(start="2019-01-01", end="2024-12-31")
    frame = validation_frame("AAPL", prices, "2020-01-01", fallback_data_used=True)

    bucket = score_bucket_analysis(frame)
    calibration = forecast_calibration(frame)
    confidence = confidence_calibration(frame)

    assert not bucket.empty
    assert "average_forward_return_pct" in bucket.columns
    assert not calibration.empty
    assert "average_actual_return_pct" in calibration.columns
    assert "forecast_bias_pct" in calibration.columns
    assert not confidence.empty
    assert "actual_hit_rate_pct" in confidence.columns


def test_feature_importance_and_model_quality_metrics():
    prices = sample_price_history(start="2019-01-01", end="2024-12-31")
    frame = validation_frame("NVDA", prices, "2020-01-01", fallback_data_used=True)

    importance = feature_importance(frame)
    quality = model_quality_metrics(frame)

    assert not importance.empty
    assert {"factor", "importance", "correlation_to_forward_return"}.issubset(
        importance.columns
    )
    assert "r_squared" in quality
    assert "mean_absolute_error_pct" in quality
    assert "rmse_pct" in quality
    assert "hit_rate_pct" in quality
    assert "sortino_ratio" in quality
    assert "information_ratio" in quality
    assert "maximum_drawdown_pct" in quality
    assert "calibration_error_pct" in quality
    assert "predictive_power" in quality
    assert "recalibration_reasons" in quality
