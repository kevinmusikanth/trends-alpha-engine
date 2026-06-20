from __future__ import annotations

import pandas as pd

from tae.backtesting.metrics import return_summary


def top_n_portfolio(
    scored_returns: pd.DataFrame,
    n: int,
    date_column: str = "date",
    score_column: str = "overall_score",
    return_column: str = "forward_return_1m",
) -> dict[str, float]:
    required = {date_column, score_column, return_column}
    missing = required - set(scored_returns.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = (
        scored_returns.sort_values([date_column, score_column], ascending=[True, False])
        .groupby(date_column)
        .head(n)
    )
    portfolio_returns = selected.groupby(date_column)[return_column].mean()
    return return_summary(portfolio_returns)

