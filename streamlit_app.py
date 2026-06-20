from __future__ import annotations

from inspect import signature

import pandas as pd
import streamlit as st

from tae.backtesting.engine import banded_forward_returns, forward_returns
from tae.connectors.fallback import sample_price_history
from tae.connectors.yahoo import YahooFinanceConnector
from tae.forecasting.backtest import prediction_test_frame, prediction_test_summary
from tae.forecasting.engine import build_forecast_report
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


def quality_for_display(score, prices: pd.DataFrame, is_fallback: bool) -> dict[str, object]:
    quality = getattr(score, "data_quality", None)
    if quality:
        return {
            "live_price_data_available": bool(quality.get("live_price_data_available")),
            "fundamental_data_available": bool(quality.get("fundamental_data_available")),
            "fallback_data_used": bool(quality.get("fallback_data_used")),
            "sample_fundamentals_used": bool(quality.get("sample_fundamentals_used")),
            "missing_metrics": list(quality.get("missing_metrics", [])),
        }

    return {
        "live_price_data_available": not prices.empty and not is_fallback,
        "fundamental_data_available": False,
        "fallback_data_used": is_fallback,
        "sample_fundamentals_used": False,
        "missing_metrics": ["fundamental data quality metadata unavailable"],
    }


def score_for_app(ticker: str, prices: pd.DataFrame, is_fallback: bool):
    parameters = signature(score_ticker).parameters
    kwargs = {}
    if "live_price_data_available" in parameters:
        kwargs["live_price_data_available"] = not is_fallback
    if "fallback_data_used" in parameters:
        kwargs["fallback_data_used"] = is_fallback

    score = score_ticker(ticker, prices, **kwargs)
    return score, quality_for_display(score, prices, is_fallback)


def forecast_frames(score, prices: pd.DataFrame):
    report = build_forecast_report(score, prices)
    forecast_frame = pd.DataFrame([line.__dict__ for line in report.forecasts])
    valuation_frame = pd.DataFrame([report.valuation.__dict__])
    positive_frame = pd.DataFrame([driver.__dict__ for driver in report.top_positive_drivers])
    negative_frame = pd.DataFrame([driver.__dict__ for driver in report.top_negative_drivers])
    exposure_frame = pd.DataFrame(
        {
            "factor": list(report.factor_exposures.keys()),
            "exposure_pct": list(report.factor_exposures.values()),
        }
    )
    return report, forecast_frame, valuation_frame, positive_frame, negative_frame, exposure_frame


def display_forecast_report(score, prices: pd.DataFrame) -> None:
    report, forecast_frame, valuation_frame, positive_frame, negative_frame, exposure_frame = (
        forecast_frames(score, prices)
    )
    st.subheader("Forecast")
    grouped = forecast_frame.groupby("group", sort=False)
    for group, frame in grouped:
        st.write(group)
        st.dataframe(
            frame[
                [
                    "horizon",
                    "expected_return_pct",
                    "bear_case_pct",
                    "base_case_pct",
                    "bull_case_pct",
                    "confidence_pct",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("Valuation")
    st.dataframe(valuation_frame, use_container_width=True)

    st.subheader("Model Explanation")
    driver_cols = st.columns(2)
    driver_cols[0].write("Top positive drivers")
    driver_cols[0].dataframe(positive_frame, use_container_width=True)
    driver_cols[1].write("Top negative drivers")
    driver_cols[1].dataframe(negative_frame, use_container_width=True)
    st.write("Factor exposures")
    st.dataframe(exposure_frame, use_container_width=True)
    st.caption(
        "Forecasts are factor-based research estimates from momentum, valuation, "
        "growth, quality, institutional, and risk factors."
    )


tab_screener, tab_forecast, tab_prediction, tab_backtest, tab_watchlist, tab_portfolio = st.tabs(
    [
        "Screener",
        "Forecast",
        "Prediction Testing",
        "Backtesting",
        "Watchlist",
        "Portfolio Testing",
    ]
)

with tab_screener:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    if st.button("Score ticker", type="primary"):
        prices, is_fallback = safe_price_history(ticker, period="1y")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        score, quality = score_for_app(ticker, prices, is_fallback)
        if quality["sample_fundamentals_used"]:
            st.warning("Sample fundamentals used")
        cols = st.columns(6)
        cols[0].metric("Short-term trading score", score.short_score)
        cols[1].metric("Medium-term alpha score", score.medium_score)
        cols[2].metric("Long-term compounder score", score.long_score)
        cols[3].metric("Risk score", score.risk_score)
        cols[4].metric("Overall score", score.overall_score)
        cols[5].metric("Label", score.recommendation)

        display_forecast_report(score, prices)

        st.subheader("Data Quality")
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


with tab_forecast:
    forecast_ticker = st.text_input("Forecast ticker", value="MSFT").upper().strip()
    if st.button("Generate forecast"):
        prices, is_fallback = safe_price_history(forecast_ticker, period="5y")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        score, quality = score_for_app(forecast_ticker, prices, is_fallback)
        if quality["sample_fundamentals_used"]:
            st.warning("Sample fundamentals used")
        score_cols = st.columns(5)
        score_cols[0].metric("Short-term trading score", score.short_score)
        score_cols[1].metric("Medium-term alpha score", score.medium_score)
        score_cols[2].metric("Long-term compounder score", score.long_score)
        score_cols[3].metric("Risk score", score.risk_score)
        score_cols[4].metric("Overall score", score.overall_score)
        display_forecast_report(score, prices)
        st.caption(ADVICE_WARNING)


with tab_prediction:
    test_ticker = st.text_input("Prediction test ticker", value="AAPL").upper().strip()
    test_start = st.text_input("Prediction test start date", value="2022-01-01")
    test_horizon = st.selectbox(
        "Prediction horizon",
        ["1 week", "1 month", "3 months", "6 months", "12 months"],
        index=1,
    )
    if st.button("Run prediction test"):
        prices, is_fallback = safe_price_history(test_ticker, start="2018-01-01")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        frame = prediction_test_frame(
            test_ticker,
            prices,
            start_date=test_start,
            horizon=test_horizon,
            fallback_data_used=is_fallback,
        )
        summary = prediction_test_summary(frame)
        metric_cols = st.columns(5)
        metric_cols[0].metric("Hit rate", f"{summary['hit_rate'] * 100:.1f}%")
        metric_cols[1].metric("Average error", f"{summary['average_error']:.1f}%")
        metric_cols[2].metric("Sharpe ratio", f"{summary['sharpe_ratio']:.2f}")
        metric_cols[3].metric("CAGR", f"{summary['cagr'] * 100:.1f}%")
        metric_cols[4].metric(
            "Maximum drawdown",
            f"{summary['maximum_drawdown'] * 100:.1f}%",
        )

        if frame.empty:
            st.warning("Not enough future price history to test this horizon.")
        else:
            chart_frame = frame.set_index("date")[["predicted_return", "actual_return"]] * 100
            st.subheader("Predicted vs Actual Returns")
            st.line_chart(chart_frame)

            accuracy = frame[["date", "hit"]].copy()
            accuracy["rolling_accuracy"] = accuracy["hit"].rolling(6).mean() * 100
            st.subheader("Rolling Accuracy")
            st.line_chart(accuracy.set_index("date")["rolling_accuracy"])

            distribution = (
                frame.groupby("score_bucket")["actual_return"]
                .mean()
                .mul(100)
                .reset_index(name="average_actual_return_pct")
            )
            st.subheader("Return Distribution by Score Bucket")
            st.bar_chart(distribution.set_index("score_bucket"))

            calibration = frame[["confidence_bucket", "hit"]].copy()
            calibration = calibration.groupby("confidence_bucket")["hit"].mean().mul(100)
            st.subheader("Confidence Calibration")
            st.bar_chart(calibration)

            st.dataframe(frame, use_container_width=True)
        st.caption(ADVICE_WARNING)


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
