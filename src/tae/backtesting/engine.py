from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import return_summary

FORWARD_WINDOWS = {
    "1w": 5,
    "2w": 10,
    "1m": 21,
    "2m": 42,
    "4m": 84,
    "6m": 126,
    "12m": 252,
}

SCORE_BANDS = [
    (90, 100, "90 to 100"),
    (80, 89.999, "80 to 89"),
    (70, 79.999, "70 to 79"),
    (60, 69.999, "60 to 69"),
    (0, 59.999, "Below 60"),
]


def score_band(score: float) -> str:
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return "Below 60"


def forward_returns(price_history: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    prices = price_history.sort_values("date").copy()
    for label, periods in FORWARD_WINDOWS.items():
        prices[f"forward_return_{label}"] = (
            prices[price_column].shift(-periods) / prices[price_column] - 1
        )
    return prices


def banded_forward_returns(score_frame: pd.DataFrame, horizon: str = "1m") -> pd.DataFrame:
    forward_col = f"forward_return_{horizon}"
    required = {"score", forward_col}
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = score_frame.copy()
    frame["score_band"] = frame["score"].apply(score_band)
    rows = []
    for band, group in frame.groupby("score_band", sort=False):
        metrics = return_summary(group[forward_col])
        rows.append(
            {"score_band": band, **metrics, "sample_size": int(group[forward_col].count())}
        )
    return pd.DataFrame(rows)
