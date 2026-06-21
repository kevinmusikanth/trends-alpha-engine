import sys
from pathlib import Path
from types import SimpleNamespace


STREAMLIT_EMPIRICAL_HELPERS = (
    "current_bucket_return_distribution",
    "empirical_fallback_message",
    "empirical_investment_outcome_table",
    "empirical_outlook_interpretation",
    "empirical_score_bucket_forecast",
    "score_bucket_comparison",
)


def load_streamlit_app():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import streamlit_app

    return streamlit_app


def test_streamlit_app_imports_successfully():
    streamlit_app = load_streamlit_app()

    assert streamlit_app.ADVICE_WARNING == "Research tool only. Not financial advice."


def test_streamlit_empirical_helpers_import_successfully():
    from tae import forecasting
    from tae.forecasting import empirical

    for helper_name in STREAMLIT_EMPIRICAL_HELPERS:
        assert callable(getattr(empirical, helper_name))
        assert callable(getattr(forecasting, helper_name))


def test_safe_price_history_falls_back_when_yahoo_fails(monkeypatch):
    streamlit_app = load_streamlit_app()

    def raise_rate_limit(*args, **kwargs):
        raise RuntimeError("Too Many Requests")

    monkeypatch.setattr(streamlit_app, "load_price_history", raise_rate_limit)

    prices, is_fallback = streamlit_app.safe_price_history("AAPL", period="1y")

    assert is_fallback is True
    assert not prices.empty
    assert {"date", "close", "volume"}.issubset(prices.columns)


def test_streamlit_scoring_path_returns_quality_metadata():
    streamlit_app = load_streamlit_app()

    prices = streamlit_app.sample_price_history(periods=260)
    score, quality = streamlit_app.score_for_app("AAPL", prices, True)
    report, forecast_frame, valuation_frame, positive_frame, negative_frame, exposure_frame = (
        streamlit_app.forecast_frames(score, prices)
    )

    assert score.ticker == "AAPL"
    assert score.long_score > 0
    assert "fallback_data_used" in quality
    assert "sample_fundamentals_used" in quality
    assert "missing_metrics" in quality
    assert not forecast_frame.empty
    assert not valuation_frame.empty
    assert not positive_frame.empty
    assert not negative_frame.empty
    assert not exposure_frame.empty
    assert report.valuation.current_price > 0


def test_streamlit_scoring_path_handles_older_score_signature(monkeypatch):
    streamlit_app = load_streamlit_app()

    def old_score_ticker(ticker, prices):
        return SimpleNamespace(
            ticker=ticker,
            short_score=1,
            medium_score=2,
            long_score=3,
            risk_score=4,
            overall_score=5,
            recommendation="Watchlist",
            components={},
        )

    monkeypatch.setattr(streamlit_app, "score_ticker", old_score_ticker)
    prices = streamlit_app.sample_price_history(periods=5)
    score, quality = streamlit_app.score_for_app("AAPL", prices, True)

    assert score.ticker == "AAPL"
    assert quality["fallback_data_used"] is True
    assert quality["fundamental_data_available"] is False


def test_screener_ticker_input_splits_comma_separated_symbols():
    streamlit_app = load_streamlit_app()

    tickers = streamlit_app.custom_tickers_from_text("AAPL, MSFT, META, NVDA")

    assert tickers == ["AAPL", "MSFT", "META", "NVDA"]


def test_screener_row_includes_empirical_metrics():
    streamlit_app = load_streamlit_app()
    score = SimpleNamespace(
        ticker="AAPL",
        overall_score=82.5,
        recommendation="Buy",
        short_score=77.0,
        medium_score=83.0,
        long_score=88.0,
        risk_score=18.0,
    )
    empirical = streamlit_app.pd.DataFrame(
        [
            {
                "horizon": "12 months",
                "average_return_pct": 24.0,
                "win_rate_pct": 72.0,
                "calibration_accuracy_pct": 80.0,
            },
            {
                "horizon": "3 years",
                "average_return_pct": 90.0,
                "win_rate_pct": 81.0,
                "calibration_accuracy_pct": 78.0,
            },
            {
                "horizon": "5 years",
                "average_return_pct": 155.0,
                "win_rate_pct": 86.0,
                "calibration_accuracy_pct": 76.0,
            },
        ]
    )

    row = streamlit_app.screener_row_from_score(score, empirical)

    assert row["ticker"] == "AAPL"
    assert row["empirical_12m_return"] == 24.0
    assert row["empirical_3y_return"] == 90.0
    assert row["empirical_5y_return"] == 155.0
    assert row["empirical_win_rate"] == 72.0
    assert row["confidence_pct"] == 78.0


def test_score_multiple_tickers_returns_sorted_screener_frame(monkeypatch):
    streamlit_app = load_streamlit_app()

    scores = {
        "AAPL": 72.0,
        "MSFT": 81.0,
        "META": 65.0,
    }

    def fake_safe_price_history(*args, **kwargs):
        return streamlit_app.sample_price_history(periods=30), True

    def fake_score_for_app(ticker, prices, is_fallback):
        score = SimpleNamespace(
            ticker=ticker,
            overall_score=scores[ticker],
            recommendation="Buy" if scores[ticker] >= 80 else "Watchlist",
            short_score=scores[ticker] - 1,
            medium_score=scores[ticker],
            long_score=scores[ticker] + 1,
            risk_score=20.0,
        )
        return score, {"fallback_data_used": is_fallback}

    monkeypatch.setattr(streamlit_app, "safe_price_history", fake_safe_price_history)
    monkeypatch.setattr(streamlit_app, "score_for_app", fake_score_for_app)

    frame = streamlit_app.score_multiple_tickers(
        ["AAPL", "MSFT", "META"],
        streamlit_app.pd.DataFrame(),
        min_observations=1,
    )

    assert frame["ticker"].tolist() == ["MSFT", "AAPL", "META"]
    assert {
        "ticker",
        "overall_score",
        "label",
        "short_term_score",
        "medium_term_score",
        "long_term_score",
        "risk_score",
        "empirical_12m_return",
        "empirical_3y_return",
        "empirical_5y_return",
        "empirical_win_rate",
        "confidence_pct",
    }.issubset(frame.columns)


def test_prediction_accuracy_dashboard_core_path_runs_with_fallback_data():
    streamlit_app = load_streamlit_app()

    prices = {
        ticker: streamlit_app.sample_price_history(start="2013-01-01", end="2026-01-01")
        for ticker in ["AAPL", "MSFT"]
    }
    flags = {ticker: True for ticker in prices}
    frame = streamlit_app.point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker=flags,
        step_days=252,
    )
    accuracy = streamlit_app.prediction_accuracy_metrics(frame)
    threshold = streamlit_app.score_threshold_validation(frame)
    calibration = streamlit_app.pit_forecast_calibration(frame)

    assert not frame.empty
    assert "average_prediction_error_pct" in accuracy
    assert not threshold.empty
    assert not calibration.empty
    assert frame["sample_fundamentals_used"].eq(False).all()


def test_streamlit_forecast_tab_empirical_core_path_runs_with_fallback_data():
    streamlit_app = load_streamlit_app()

    prices = streamlit_app.sample_price_history(periods=300)
    score, quality = streamlit_app.score_for_app("META", prices, True)
    report, *_ = streamlit_app.forecast_frames(score, prices)
    validation_records = streamlit_app.point_in_time_prediction_frame(
        {
            ticker: streamlit_app.sample_price_history(
                start="2013-01-01",
                end="2026-01-01",
            )
            for ticker in ["AAPL", "MSFT", "META"]
        },
        start_date="2016-01-01",
        fallback_data_used_by_ticker={"AAPL": True, "MSFT": True, "META": True},
        step_days=252,
    )
    empirical = streamlit_app.empirical_score_bucket_forecast(
        score.overall_score,
        validation_records,
        investment_amount=10,
        min_observations=1,
    )
    outlook = streamlit_app.empirical_outlook_interpretation(empirical)
    outcome = streamlit_app.empirical_investment_outcome_table(empirical)

    assert quality["fallback_data_used"] is True
    assert report.ticker == "META"
    assert not validation_records.empty
    assert not empirical.empty
    assert "expected_value" in empirical.columns
    assert outlook["headline"].startswith("Empirical Outlook:")
    assert not outcome.empty
