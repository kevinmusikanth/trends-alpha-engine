import pandas as pd
import pytest

from tae.connectors.fallback import sample_price_history
from tae.forecasting.backtest import (
    actual_forward_return,
    prediction_test_frame,
    prediction_test_summary,
)
from tae.forecasting.engine import build_forecast_report
from tae.scoring.engine import score_ticker


def test_forecast_generation_has_grouped_bear_base_bull_cases():
    prices = sample_price_history(periods=300)
    score = score_ticker("NVDA", prices, fallback_data_used=True)
    report = build_forecast_report(score, prices)

    horizons = {line.horizon for line in report.forecasts}
    assert {"1 week", "1 month", "3 months", "6 months", "12 months"}.issubset(horizons)
    assert {"3 year CAGR", "5 year CAGR"}.issubset(horizons)
    assert report.valuation.current_price > 0
    assert 0 <= report.valuation.probability_positive_return_pct <= 100
    for line in report.forecasts:
        assert line.bear_case_pct <= line.base_case_pct <= line.bull_case_pct
        assert 0 <= line.confidence_pct <= 100


def test_forecast_explanation_has_drivers_and_data_quality():
    prices = sample_price_history(periods=300)
    score = score_ticker("MSFT", prices, fallback_data_used=True)
    report = build_forecast_report(score, prices)

    assert report.top_positive_drivers
    assert report.top_negative_drivers
    assert "fallback_data_used" in report.data_quality
    assert "sample_fundamentals_used" in report.data_quality


def test_actual_forward_return_uses_requested_horizon():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="B"),
            "close": [100 + index for index in range(40)],
        }
    )
    actual = actual_forward_return(prices, pd.Timestamp("2024-01-01"), "1 week")
    assert actual == pytest.approx(0.05)


def test_prediction_testing_and_summary_work_in_fallback_mode():
    prices = sample_price_history(start="2020-01-01", end="2024-12-31")
    frame = prediction_test_frame(
        "AAPL",
        prices,
        start_date="2021-01-01",
        horizon="1 month",
        fallback_data_used=True,
    )
    summary = prediction_test_summary(frame)

    assert not frame.empty
    assert {"predicted_return", "actual_return", "error_pct", "hit"}.issubset(frame.columns)
    assert "score_bucket" in frame.columns
    assert 0 <= summary["hit_rate"] <= 1
    assert "maximum_drawdown" in summary

