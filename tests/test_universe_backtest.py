import pytest

from tae.connectors.fallback import sample_price_history
from tae.forecasting.universe import (
    UNIVERSE_HORIZONS,
    score_bucket_20_point,
    universe_bucket_summary,
    universe_calibration_curve,
    universe_prediction_frame,
)
from tae.universe import NASDAQ100_SAMPLE, SP500_SAMPLE, get_universe


@pytest.mark.parametrize(
    ("score", "bucket"),
    [
        (0, "0-20"),
        (19.99, "0-20"),
        (20, "20-40"),
        (40, "40-60"),
        (60, "60-80"),
        (80, "80-100"),
        (100, "80-100"),
    ],
)
def test_score_bucket_20_point(score, bucket):
    assert score_bucket_20_point(score) == bucket


def test_universe_prediction_frame_builds_all_required_horizons():
    tickers = ["AAPL", "MSFT", "NVDA"]
    price_history = {
        ticker: sample_price_history(start="2020-01-01", end="2024-12-31")
        for ticker in tickers
    }
    fallback_flags = {ticker: True for ticker in tickers}

    frame = universe_prediction_frame(
        price_history,
        start_date="2022-01-01",
        fallback_data_used_by_ticker=fallback_flags,
    )

    assert not frame.empty
    assert set(UNIVERSE_HORIZONS).issubset(set(frame["horizon"]))
    assert {"ticker", "score_bucket", "predicted_return", "actual_return"}.issubset(
        frame.columns
    )
    assert set(frame["ticker"]) == set(tickers)


def test_universe_bucket_summary_reports_validation_metrics():
    price_history = {
        ticker: sample_price_history(start="2020-01-01", end="2024-12-31")
        for ticker in ["AAPL", "MSFT", "NVDA"]
    }
    frame = universe_prediction_frame(
        price_history,
        start_date="2022-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in price_history},
    )

    summary = universe_bucket_summary(frame)
    calibration = universe_calibration_curve(frame)

    assert not summary.empty
    assert {
        "average_return_pct",
        "median_return_pct",
        "win_rate_pct",
        "sharpe_ratio",
        "maximum_drawdown_pct",
    }.issubset(summary.columns)
    assert not calibration.empty
    assert {
        "average_predicted_return_pct",
        "average_actual_return_pct",
        "forecast_bias_pct",
    }.issubset(calibration.columns)


def test_public_universe_loader_falls_back_when_live_constituents_are_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("pandas.read_html", unavailable)

    assert get_universe("sp500") == SP500_SAMPLE
    assert get_universe("nasdaq100") == NASDAQ100_SAMPLE
