import pandas as pd
import pytest

from tae.backtesting.engine import banded_forward_returns, forward_returns, score_band
from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate


def test_score_band_labels():
    assert score_band(95) == "90 to 100"
    assert score_band(85) == "80 to 89"
    assert score_band(75) == "70 to 79"
    assert score_band(65) == "60 to 69"
    assert score_band(40) == "Below 60"


def test_forward_returns_adds_windows():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=260, freq="B"),
            "close": [100 + i for i in range(260)],
        }
    )
    result = forward_returns(prices)
    assert "forward_return_1w" in result.columns
    assert result.loc[0, "forward_return_1w"] == pytest.approx(0.05)


def test_banded_forward_returns():
    frame = pd.DataFrame({"score": [95, 85, 75], "forward_return_1m": [0.2, -0.1, 0.05]})
    summary = banded_forward_returns(frame, horizon="1m")
    assert set(summary["score_band"]) == {"90 to 100", "80 to 89", "70 to 79"}


def test_metrics_handle_empty_and_zero_variance():
    empty = pd.Series(dtype=float)
    assert win_rate(empty) == 0
    assert max_drawdown(empty) == 0
    assert sharpe_ratio(pd.Series([0.01, 0.01])) == 0
