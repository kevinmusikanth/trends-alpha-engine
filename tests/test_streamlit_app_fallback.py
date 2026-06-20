import sys
from pathlib import Path
from types import SimpleNamespace


def load_streamlit_app():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import streamlit_app

    return streamlit_app


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
