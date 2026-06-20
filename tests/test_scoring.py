import pandas as pd

from tae.scoring.engine import score_ticker
from tae.scoring.models import MODEL_A_WEIGHTS, MODEL_B_WEIGHTS, MODEL_C_WEIGHTS


def test_model_weights_sum_to_100():
    assert sum(MODEL_A_WEIGHTS.values()) == 100
    assert sum(MODEL_B_WEIGHTS.values()) == 100
    assert sum(MODEL_C_WEIGHTS.values()) == 100


def test_score_ticker_handles_missing_data():
    score = score_ticker("ABC", pd.DataFrame())
    assert score.ticker == "ABC"
    assert 0 <= score.overall_score <= 100
    assert score.recommendation in {"Buy", "Watch", "Avoid"}


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

