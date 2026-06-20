from __future__ import annotations

from datetime import date

import pandas as pd


def sample_price_history(
    start: str | date | None = None,
    end: str | date | None = None,
    periods: int = 260,
) -> pd.DataFrame:
    """Create deterministic fallback OHLCV data when live data is unavailable."""
    if start:
        dates = pd.date_range(start=start, end=end or pd.Timestamp.today().normalize(), freq="B")
    elif end:
        dates = pd.date_range(end=end, periods=periods, freq="B")
    else:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")

    if len(dates) == 0:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")

    rows = []
    for index, current_date in enumerate(dates):
        trend = 100 + (index * 0.18)
        seasonal = ((index % 21) - 10) * 0.12
        close = round(trend + seasonal, 2)
        rows.append(
            {
                "date": current_date,
                "open": round(close * 0.995, 2),
                "high": round(close * 1.012, 2),
                "low": round(close * 0.988, 2),
                "close": close,
                "adj_close": close,
                "volume": 1_000_000 + ((index % 30) * 20_000),
            }
        )
    return pd.DataFrame(rows)
