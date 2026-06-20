import pandas as pd
import pytest

from tae.connectors.fallback import sample_price_history
from tae.forecast.engine import FORECAST_HORIZONS, forecast_from_score
from tae.forecast.testing import (
    actual_forward_return,
    prediction_test_frame,
    prediction_test_summary,
)
from tae.scoring.engine import score_ticker


def test_forecast_returns_percent_ranges_for_all_horizons():
    prices = sample_price_history(periods=300)
    score = score_ticker("MSFT", prices, fallback_data_used=True)
    forecast = forecast_from_score(score, prices)

    assert len(forecast.rows) == len(FORECAST_HORIZONS)
    assert 0 <= forecast.confidence_score <= 100
    assert set(forecast.factor_exposures) == {"momentum", "valuation", "growth", "quality"}
    for row in forecast.rows:
        assert row["bear_case_return"] <= row["base_case_return"]
        assert row["base_case_return"] <= row["bull_case_return"]


def test_actual_forward_return_uses_future_price():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="B"),
            "close": [100 + index for index in range(40)],
        }
    )
    actual = actual_forward_return(prices, pd.Timestamp("2024-01-01"), 5)
    assert actual == pytest.approx(0.05)


def test_prediction_test_frame_and_summary_work_with_fallback_data():
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
    assert {"forecast_return", "actual_return", "confidence_score"}.issubset(frame.columns)
    assert 0 <= summary["hit_rate"] <= 1
    assert "maximum_drawdown" in summary
