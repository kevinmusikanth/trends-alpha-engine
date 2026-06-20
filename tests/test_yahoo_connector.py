import pytest

from tae.connectors.fallback import sample_price_history
from tae.connectors.yahoo import YahooFinanceConnector


def test_sample_price_history_has_expected_columns():
    prices = sample_price_history(periods=5)
    assert list(prices.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(prices) == 5


def test_yahoo_retry_raises_runtime_error_after_failures():
    connector = YahooFinanceConnector(max_retries=2, backoff_seconds=0)

    with pytest.raises(RuntimeError, match="Yahoo Finance request failed"):
        connector._retry(lambda: (_ for _ in ()).throw(RuntimeError("rate limited")))
