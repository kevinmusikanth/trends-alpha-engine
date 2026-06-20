import sys
from pathlib import Path


def test_safe_price_history_falls_back_when_yahoo_fails(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import streamlit_app

    def raise_rate_limit(*args, **kwargs):
        raise RuntimeError("Too Many Requests")

    monkeypatch.setattr(streamlit_app, "load_price_history", raise_rate_limit)

    prices, is_fallback = streamlit_app.safe_price_history("AAPL", period="1y")

    assert is_fallback is True
    assert not prices.empty
    assert {"date", "close", "volume"}.issubset(prices.columns)
