import pandas as pd
import pytest

from tae.connectors.fallback import sample_price_history
from tae.scoring.engine import score_ticker
from tae.scoring.fundamentals import FUNDAMENTAL_FEATURE_KEYS, SAMPLE_FUNDAMENTALS
from tae.scoring.models import MODEL_A_WEIGHTS, MODEL_B_WEIGHTS, MODEL_C_WEIGHTS


def test_model_weights_sum_to_100():
    assert sum(MODEL_A_WEIGHTS.values()) == 100
    assert sum(MODEL_B_WEIGHTS.values()) == 100
    assert sum(MODEL_C_WEIGHTS.values()) == 100


def test_score_ticker_handles_missing_data():
    score = score_ticker("ABC", pd.DataFrame())
    assert score.ticker == "ABC"
    assert 0 <= score.overall_score <= 100
    assert score.recommendation in {"Strong Buy", "Buy", "Watchlist", "Avoid"}
    assert score.data_quality["fundamental_data_available"] is False


def test_score_ticker_uses_price_momentum():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=130, freq="B"),
            "close": range(100, 230),
            "volume": [1_000_000] * 130,
        }
    )
    score = score_ticker("XYZ", prices)
    assert score.short_score > 0
    assert score.medium_score > 0


@pytest.mark.parametrize(
    "ticker",
    [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "BRK-B",
        "JPM",
        "COST",
        "TSLA",
        "PLTR",
    ],
)
def test_known_research_tickers_use_sample_fundamentals(ticker):
    prices = sample_price_history(periods=260)
    score = score_ticker(
        ticker,
        prices,
        live_price_data_available=False,
        fallback_data_used=True,
    )

    assert score.short_score > 20
    assert score.medium_score > 20
    assert score.long_score > 20
    assert score.overall_score > 20
    assert score.data_quality["sample_fundamentals_used"] is True
    assert score.data_quality["fundamental_data_available"] is True


def test_meta_uses_sample_fundamentals_and_scores_are_not_zero():
    prices = sample_price_history(periods=260)
    score = score_ticker(
        "META",
        prices,
        live_price_data_available=False,
        fallback_data_used=True,
    )

    assert score.medium_score > 20
    assert score.long_score > 20
    assert score.overall_score > 20
    assert score.recommendation in {"Strong Buy", "Buy", "Watchlist", "Avoid"}
    assert score.data_quality["sample_fundamentals_used"] is True
    assert score.data_quality["fundamental_data_available"] is True
    assert not any(
        metric in score.data_quality["missing_metrics"]
        for metric in FUNDAMENTAL_FEATURE_KEYS
    )


def test_requested_mega_cap_sample_fundamentals_are_complete():
    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "BRK-B",
        "JPM",
        "COST",
        "TSLA",
        "PLTR",
    ]

    for ticker in tickers:
        assert ticker in SAMPLE_FUNDAMENTALS
        assert FUNDAMENTAL_FEATURE_KEYS.issubset(SAMPLE_FUNDAMENTALS[ticker])
