from __future__ import annotations

import pandas as pd
import streamlit as st

from tae.backtesting.engine import banded_forward_returns, forward_returns
from tae.connectors.fallback import sample_price_history
from tae.connectors.yahoo import YahooFinanceConnector
from tae.scoring.engine import score_ticker
from tae.universe import get_universe

st.set_page_config(page_title="Trends Alpha Engine", layout="wide")

st.title("Trends Alpha Engine")
ADVICE_WARNING = "Research tool only. Not financial advice."
st.caption(ADVICE_WARNING)

YAHOO_WARNING = "Live Yahoo data temporarily unavailable."


def yes_no(value: object) -> str:
    return "Yes" if bool(value) else "No"


@st.cache_data(ttl=900, show_spinner=False)
def load_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    connector = YahooFinanceConnector(max_retries=3, backoff_seconds=1.5)
    return connector.fetch_price_history(ticker, start=start, end=end, period=period)


def safe_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> tuple[pd.DataFrame, bool]:
    try:
        prices = load_price_history(ticker, start=start, end=end, period=period)
        if not prices.empty:
            return prices, False
    except Exception:
        pass

    return sample_price_history(start=start, end=end), True

tab_screener, tab_backtest, tab_watchlist, tab_portfolio = st.tabs(
    ["Screener", "Backtesting", "Watchlist", "Portfolio Testing"]
)

with tab_screener:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    if st.button("Score ticker", type="primary"):
        prices, is_fallback = safe_price_history(ticker, period="1y")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        score = score_ticker(
            ticker,
            prices,
            live_price_data_available=not is_fallback,
            fallback_data_used=is_fallback,
        )
        if score.data_quality["sample_fundamentals_used"]:
            st.warning("Sample fundamentals used")
        cols = st.columns(6)
        cols[0].metric("Short-term trading score", score.short_score)
        cols[1].metric("Medium-term alpha score", score.medium_score)
        cols[2].metric("Long-term compounder score", score.long_score)
        cols[3].metric("Risk score", score.risk_score)
        cols[4].metric("Overall score", score.overall_score)
        cols[5].metric("Label", score.recommendation)

        st.subheader("Data Quality")
        quality = score.data_quality
        quality_cols = st.columns(4)
        quality_cols[0].metric(
            "Live price data available",
            yes_no(quality["live_price_data_available"]),
        )
        quality_cols[1].metric(
            "Fundamental data available",
            yes_no(quality["fundamental_data_available"]),
        )
        quality_cols[2].metric("Fallback data used", yes_no(quality["fallback_data_used"]))
        quality_cols[3].metric(
            "Sample fundamentals used",
            yes_no(quality["sample_fundamentals_used"]),
        )

        missing = quality["missing_metrics"]
        if missing:
            st.write("Missing metrics")
            st.dataframe(pd.DataFrame({"metric": missing}), use_container_width=True)
        else:
            st.success("No missing metrics for this score.")

        st.subheader("Component Scores")
        for model_name, components in score.components.items():
            st.write(model_name.replace("_", " ").title())
            st.dataframe(pd.DataFrame(components), use_container_width=True)

with tab_backtest:
    bt_ticker = st.text_input("Backtest ticker", value="MSFT").upper().strip()
    start = st.text_input("Start date", value="2021-01-01")
    end = st.text_input("End date", value="2024-12-31")
    horizon = st.selectbox(
        "Forward return horizon",
        ["1w", "2w", "1m", "2m", "4m", "6m", "12m"],
        index=2,
    )
    if st.button("Run backtest"):
        prices, is_fallback = safe_price_history(bt_ticker, start=start, end=end)
        if is_fallback:
            st.warning(YAHOO_WARNING)
        st.caption(ADVICE_WARNING)
        returns = forward_returns(prices)
        returns["score"] = 50.0
        st.dataframe(banded_forward_returns(returns, horizon=horizon), use_container_width=True)
        st.line_chart(returns.set_index("date")["close"])

with tab_watchlist:
    st.info(
        "Version 1 includes watchlist data structures and score-change labels. "
        "Database-backed watchlists are the next implementation step."
    )

with tab_portfolio:
    universe_name = st.selectbox("Universe", ["sp500", "nasdaq100", "russell2000"])
    n = st.selectbox("Portfolio size", [10, 25, 50])
    st.write("Sample universe tickers:", ", ".join(get_universe(universe_name)))
    st.info(f"Top {n} hypothetical portfolio testing is scaffolded in `tae.portfolio`.")
