from __future__ import annotations

from inspect import signature

import pandas as pd
import streamlit as st

from tae.backtesting.engine import banded_forward_returns, forward_returns
from tae.connectors.fallback import sample_price_history
from tae.connectors.yahoo import YahooFinanceConnector
from tae.forecasting.alpha_validation import (
    alpha_validation_frame,
    benchmark_comparison,
    benchmark_return_frame,
    confidence_framework,
    final_investment_outcome_card,
    investment_outcome_by_bucket,
    predictor_validation_metrics,
    score_bucket_performance,
    threshold_analysis,
)
from tae.forecasting.backtest import prediction_test_frame, prediction_test_summary
from tae.forecasting.engine import build_forecast_report
from tae.forecasting.empirical import (
    current_bucket_return_distribution,
    empirical_fallback_message,
    empirical_investment_outcome_table,
    empirical_outlook_interpretation,
    empirical_score_bucket_forecast,
    score_bucket_comparison,
)
from tae.forecasting.outcomes import investment_outcome_projection, outcome_growth_paths
from tae.forecasting.point_in_time import (
    forecast_calibration as pit_forecast_calibration,
    investment_outcome_validation,
    point_in_time_prediction_frame,
    prediction_accuracy_metrics,
    score_threshold_validation,
)
from tae.forecasting.universe import (
    universe_bucket_summary,
    universe_calibration_curve,
    universe_prediction_frame,
)
from tae.forecasting.validation import (
    confidence_calibration,
    feature_importance,
    forecast_calibration,
    model_quality_metrics,
    score_bucket_analysis,
    validation_frame,
)
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


def format_money(value: float, currency: str) -> str:
    return f"{currency}{value:,.0f}"


def display_investment_outcome(report) -> None:
    st.subheader("Investment Outcome Projection")
    controls = st.columns([2, 1])
    investment_amount = controls[0].number_input(
        "Investment amount",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        key=f"investment_amount_{report.ticker}",
    )
    currency = controls[1].selectbox(
        "Currency",
        ["$", "£", "€", "R"],
        key=f"investment_currency_{report.ticker}",
    )

    outcomes = investment_outcome_projection(report, investment_amount)
    st.write(f"If you invest {format_money(investment_amount, currency)} today:")
    for row in outcomes.to_dict("records"):
        st.write(row["horizon"])
        case_cols = st.columns(3)
        case_cols[0].metric("Bear", format_money(row["bear_case_value"], currency))
        case_cols[1].metric("Base", format_money(row["base_case_value"], currency))
        case_cols[2].metric("Bull", format_money(row["bull_case_value"], currency))

    display_frame = outcomes.copy()
    for column in [
        "bear_case_value",
        "base_case_value",
        "bull_case_value",
        "expected_value",
    ]:
        display_frame[column] = display_frame[column].map(
            lambda value: format_money(value, currency)
        )
    st.dataframe(
        display_frame[
            [
                "horizon",
                "bear_case_value",
                "base_case_value",
                "bull_case_value",
                "expected_value",
                "expected_cagr_pct",
                "probability_positive_return_pct",
                "probability_losing_money_pct",
            ]
        ],
        use_container_width=True,
    )

    growth_paths = outcome_growth_paths(outcomes, investment_amount)
    chart = growth_paths.set_index("horizon")[["Bear", "Base", "Bull"]]
    st.subheader("Projected Growth Paths")
    st.line_chart(chart)


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

    display_investment_outcome(report)

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


def custom_tickers_from_text(raw_tickers: str) -> list[str]:
    tickers = []
    for value in raw_tickers.replace("\n", ",").split(","):
        ticker = value.upper().strip()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def load_price_histories_for_tickers(
    tickers: list[str],
    start: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    price_history_by_ticker = {}
    fallback_flags = {}
    progress = st.progress(0)
    for index, ticker in enumerate(tickers, start=1):
        prices, is_fallback = safe_price_history(ticker, start=start)
        price_history_by_ticker[ticker] = prices
        fallback_flags[ticker] = is_fallback
        progress.progress(index / len(tickers))
    return price_history_by_ticker, fallback_flags


@st.cache_data(ttl=1800, show_spinner=False)
def empirical_validation_records(
    tickers: tuple[str, ...],
    start_date: str,
    price_start: str,
    step_days: int,
) -> pd.DataFrame:
    price_history_by_ticker = {}
    fallback_flags = {}
    for ticker in tickers:
        prices, is_fallback = safe_price_history(ticker, start=price_start)
        price_history_by_ticker[ticker] = prices
        fallback_flags[ticker] = is_fallback
    return point_in_time_prediction_frame(
        price_history_by_ticker,
        start_date=start_date,
        fallback_data_used_by_ticker=fallback_flags,
        step_days=step_days,
    )


def display_empirical_forecast_section(
    score,
    report,
    validation_records: pd.DataFrame,
    investment_amount: float,
    currency: str,
    min_observations: int,
) -> None:
    empirical = empirical_score_bucket_forecast(
        score.overall_score,
        validation_records,
        investment_amount=investment_amount,
        min_observations=min_observations,
    )
    theoretical = pd.DataFrame([line.__dict__ for line in report.forecasts])
    theoretical = theoretical[["horizon", "base_case_pct", "confidence_pct"]].copy()
    theoretical["horizon"] = theoretical["horizon"].replace(
        {"3 year CAGR": "3 years", "5 year CAGR": "5 years"}
    )

    st.subheader("Historical Outcome Forecast")
    st.info(
        "This forecast is based on historical point-in-time outcomes of stocks "
        "that scored in the same range, not on today's data alone."
    )
    st.write(empirical_fallback_message(empirical))

    outlook = empirical_outlook_interpretation(empirical)
    score_bucket = empirical["score_bucket"].iloc[0] if not empirical.empty else "Unknown"
    st.subheader("Empirical Outlook")
    outlook_cols = st.columns(4)
    outlook_cols[0].metric("Current score", f"{score.overall_score:.2f}")
    outlook_cols[1].metric("Score bucket", score_bucket)
    outlook_cols[2].metric("Confidence rating", outlook["confidence"])
    outlook_cols[3].metric(
        "Observations",
        int(empirical["observation_count"].max()) if not empirical.empty else 0,
    )
    st.success(outlook["headline"])
    st.write(outlook["explanation"])
    if outlook["evidence"]:
        st.write("Historical evidence")
        st.write(pd.DataFrame({"Evidence": outlook["evidence"]}))

    st.subheader("Empirical Investment Outcome")
    st.caption(
        "Expected values use historical average returns from the current score bucket, "
        "not theoretical forecast percentages."
    )
    outcome_display = empirical_investment_outcome_table(
        empirical,
        investment_amount=investment_amount,
        currency=currency,
    )
    if outcome_display.empty:
        st.warning("No empirical investment outcomes available for this score bucket.")
    else:
        outcome_display = outcome_display.copy()
        outcome_display["expected_value"] = outcome_display["expected_value"].map(
            lambda value: format_money(value, currency)
        )
        st.dataframe(outcome_display, use_container_width=True)

    compare_cols = st.columns(2)
    compare_cols[0].write("Theoretical Forecast")
    compare_cols[0].dataframe(theoretical, use_container_width=True)
    compare_cols[1].write("Empirical Score-Bucket Forecast")
    empirical_display = empirical.copy()
    empirical_display["expected_value"] = empirical_display["expected_value"].map(
        lambda value: format_money(value, currency)
    )
    compare_cols[1].dataframe(empirical_display, use_container_width=True)

    chart_cols = st.columns(3)
    chart_cols[0].write("Win Rate by Horizon")
    chart_cols[0].bar_chart(empirical.set_index("horizon")["win_rate_pct"])
    chart_cols[1].write(f"{format_money(investment_amount, currency)} Outcome")
    chart_cols[1].bar_chart(empirical.set_index("horizon")["expected_value"])
    distribution = current_bucket_return_distribution(
        score.overall_score,
        validation_records,
        horizon="12 months",
    )
    chart_cols[2].write("12M Return Distribution")
    if distribution.empty:
        chart_cols[2].warning("No 12-month observations for this bucket.")
    else:
        chart_cols[2].bar_chart(distribution.reset_index(drop=True))

    comparison = score_bucket_comparison(validation_records)
    if not comparison.empty:
        st.subheader("Score Bucket Comparison")
        comparison_horizon = st.selectbox(
            "Empirical comparison horizon",
            ["1 week", "1 month", "3 months", "6 months", "12 months", "3 years", "5 years"],
        )
        comparison_chart = comparison[comparison["horizon"] == comparison_horizon]
        st.bar_chart(comparison_chart.set_index("score_bucket")["average_return_pct"])


(
    tab_screener,
    tab_forecast,
    tab_prediction,
    tab_validation,
    tab_prediction_accuracy,
    tab_alpha_validation,
    tab_universe,
    tab_backtest,
    tab_watchlist,
    tab_portfolio,
) = st.tabs(
    [
        "Screener",
        "Forecast",
        "Prediction Testing",
        "Validation Dashboard",
        "Prediction Accuracy Dashboard",
        "ALPHA VALIDATION",
        "Universe Backtest",
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
        if not quality["fundamental_data_available"]:
            st.warning("Fundamental data missing — score incomplete")
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
    empirical_controls = st.expander("Empirical forecast settings")
    with empirical_controls:
        empirical_universe_text = st.text_area(
            "Empirical validation tickers",
            value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, COST, TSLA, PLTR",
        )
        empirical_start = st.text_input("Empirical validation start date", value="2016-01-01")
        empirical_step = st.number_input(
            "Empirical historical date step",
            min_value=21,
            max_value=252,
            value=63,
            step=21,
        )
        empirical_min_observations = st.number_input(
            "Minimum empirical observations",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
        )
        empirical_investment = st.number_input(
            "Empirical investment amount",
            min_value=0.0,
            value=10.0,
            step=10.0,
        )
        empirical_currency = st.selectbox(
            "Empirical currency",
            ["$", "£", "€", "R"],
        )
    if st.button("Generate forecast"):
        prices, is_fallback = safe_price_history(forecast_ticker, period="5y")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        score, quality = score_for_app(forecast_ticker, prices, is_fallback)
        if quality["sample_fundamentals_used"]:
            st.warning("Sample fundamentals used")
        if not quality["fundamental_data_available"]:
            st.warning("Fundamental data missing — score incomplete")
        score_cols = st.columns(5)
        score_cols[0].metric("Short-term trading score", score.short_score)
        score_cols[1].metric("Medium-term alpha score", score.medium_score)
        score_cols[2].metric("Long-term compounder score", score.long_score)
        score_cols[3].metric("Risk score", score.risk_score)
        score_cols[4].metric("Overall score", score.overall_score)
        report, forecast_frame, valuation_frame, positive_frame, negative_frame, exposure_frame = (
            forecast_frames(score, prices)
        )
        display_forecast_report(score, prices)
        empirical_tickers = tuple(custom_tickers_from_text(empirical_universe_text))
        if empirical_tickers:
            validation_records = empirical_validation_records(
                empirical_tickers,
                start_date=empirical_start,
                price_start="2013-01-01",
                step_days=int(empirical_step),
            )
            display_empirical_forecast_section(
                score,
                report,
                validation_records,
                investment_amount=empirical_investment,
                currency=empirical_currency,
                min_observations=int(empirical_min_observations),
            )
        else:
            st.warning("Add empirical validation tickers to build a historical outcome forecast.")
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


with tab_validation:
    validation_ticker = st.text_input("Validation ticker", value="MSFT").upper().strip()
    validation_start = st.text_input("Validation start date", value="2020-01-01")
    if st.button("Run model validation"):
        prices, is_fallback = safe_price_history(validation_ticker, start="2018-01-01")
        if is_fallback:
            st.warning(YAHOO_WARNING)
        frame = validation_frame(
            validation_ticker,
            prices,
            start_date=validation_start,
            fallback_data_used=is_fallback,
        )
        quality = model_quality_metrics(frame)
        metric_cols = st.columns(5)
        metric_cols[0].metric("R squared", f"{quality['r_squared']:.3f}")
        metric_cols[1].metric("MAE", f"{quality['mean_absolute_error_pct']:.1f}%")
        metric_cols[2].metric("RMSE", f"{quality['rmse_pct']:.1f}%")
        metric_cols[3].metric("Hit rate", f"{quality['hit_rate_pct']:.1f}%")
        metric_cols[4].metric("Sharpe", f"{quality['sharpe_ratio']:.2f}")

        metric_cols_2 = st.columns(4)
        metric_cols_2[0].metric("Sortino", f"{quality['sortino_ratio']:.2f}")
        metric_cols_2[1].metric(
            "Information ratio",
            f"{quality['information_ratio']:.2f}",
        )
        metric_cols_2[2].metric(
            "Maximum drawdown",
            f"{quality['maximum_drawdown_pct']:.1f}%",
        )
        metric_cols_2[3].metric(
            "Calibration error",
            f"{quality['calibration_error_pct']:.1f}%",
        )

        st.metric("Predictive power", quality["predictive_power"])
        if quality["recalibration_flag"]:
            st.warning(quality["recalibration_flag"])
            st.write("Reasons")
            st.write(quality["recalibration_reasons"])

        if frame.empty:
            st.warning("Not enough validation history for this ticker/date.")
        else:
            bucket_frame = score_bucket_analysis(frame)
            calibration_frame = forecast_calibration(frame)
            confidence_frame = confidence_calibration(frame)
            importance_frame = feature_importance(frame)

            st.subheader("Score Bucket Analysis")
            st.dataframe(bucket_frame, use_container_width=True)
            if not bucket_frame.empty:
                chart = bucket_frame.pivot(
                    index="score_bucket",
                    columns="horizon",
                    values="average_forward_return_pct",
                )
                st.subheader("Score Bucket Returns")
                st.bar_chart(chart)

            st.subheader("Predicted vs Actual")
            st.line_chart(
                frame.set_index("date")[["predicted_return", "actual_return"]] * 100
            )

            st.subheader("Forecast Calibration")
            st.dataframe(calibration_frame, use_container_width=True)
            if not calibration_frame.empty:
                calibration_chart = calibration_frame.pivot(
                    index="forecast_bucket",
                    columns="horizon",
                    values="average_actual_return_pct",
                )
                st.subheader("Calibration Curve")
                st.line_chart(calibration_chart)

            st.subheader("Rolling Model Accuracy")
            rolling = frame[["date", "hit"]].copy()
            rolling["rolling_accuracy_pct"] = rolling["hit"].rolling(8).mean() * 100
            st.line_chart(rolling.set_index("date")["rolling_accuracy_pct"])

            st.subheader("Confidence Calibration")
            st.dataframe(confidence_frame, use_container_width=True)
            if not confidence_frame.empty:
                confidence_chart = confidence_frame.pivot(
                    index="confidence_bucket",
                    columns="horizon",
                    values="actual_hit_rate_pct",
                )
                st.bar_chart(confidence_chart)

            st.subheader("Feature Importance")
            st.dataframe(importance_frame, use_container_width=True)
            if not importance_frame.empty:
                st.bar_chart(importance_frame.set_index("factor")["importance"])
        st.caption(ADVICE_WARNING)


with tab_prediction_accuracy:
    pit_universe_choice = st.selectbox(
        "Point-in-time universe",
        ["S&P 500", "Nasdaq 100", "Custom tickers"],
    )
    pit_custom_universe = ""
    if pit_universe_choice == "Custom tickers":
        pit_custom_universe = st.text_area(
            "Point-in-time custom tickers",
            value="AAPL, MSFT, NVDA, AMZN, META",
        )
    pit_start = st.text_input("Point-in-time start date", value="2018-01-01")
    pit_max_tickers = st.number_input(
        "Point-in-time maximum tickers",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )
    pit_step_days = st.number_input(
        "Point-in-time historical date step",
        min_value=1,
        max_value=252,
        value=21,
        step=1,
    )
    pit_investment_cols = st.columns([2, 1])
    pit_investment_amount = pit_investment_cols[0].number_input(
        "Point-in-time investment amount",
        min_value=0.0,
        value=10.0,
        step=10.0,
    )
    pit_currency = pit_investment_cols[1].selectbox(
        "Point-in-time currency",
        ["$", "£", "€", "R"],
    )

    if st.button("Run point-in-time validation"):
        if pit_universe_choice == "S&P 500":
            pit_tickers = get_universe("sp500")
        elif pit_universe_choice == "Nasdaq 100":
            pit_tickers = get_universe("nasdaq100")
        else:
            pit_tickers = custom_tickers_from_text(pit_custom_universe)

        pit_tickers = pit_tickers[: int(pit_max_tickers)]
        if not pit_tickers:
            st.warning("Add at least one ticker to run point-in-time validation.")
        else:
            pit_prices, pit_fallback_flags = load_price_histories_for_tickers(
                pit_tickers,
                start="2013-01-01",
            )
            if any(pit_fallback_flags.values()):
                st.warning(YAHOO_WARNING)
            st.info(
                "Point-in-time mode disables sample fundamentals so historical scores "
                "are reconstructed from data available up to each test date."
            )

            pit_frame = point_in_time_prediction_frame(
                pit_prices,
                start_date=pit_start,
                fallback_data_used_by_ticker=pit_fallback_flags,
                step_days=int(pit_step_days),
            )
            accuracy = prediction_accuracy_metrics(pit_frame)
            threshold_frame = score_threshold_validation(pit_frame)
            calibration_frame = pit_forecast_calibration(pit_frame)

            spy_prices, _ = safe_price_history("SPY", start="2013-01-01")
            qqq_prices, _ = safe_price_history("QQQ", start="2013-01-01")
            pit_benchmarks = {
                "S&P 500": benchmark_return_frame(
                    "S&P 500",
                    spy_prices,
                    start_date=pit_start,
                    step_days=int(pit_step_days),
                ),
                "Nasdaq 100": benchmark_return_frame(
                    "Nasdaq 100",
                    qqq_prices,
                    start_date=pit_start,
                    step_days=int(pit_step_days),
                ),
            }
            outcome_frame = investment_outcome_validation(
                pit_frame,
                pit_investment_amount,
                benchmarks=pit_benchmarks,
            )

            st.subheader("Prediction Accuracy Dashboard")
            coverage = st.columns(4)
            coverage[0].metric("Tickers", len(pit_tickers))
            coverage[1].metric("Observations", len(pit_frame))
            coverage[2].metric(
                "Fallback tickers",
                sum(1 for value in pit_fallback_flags.values() if value),
            )
            coverage[3].metric(
                "Sample fundamentals used",
                yes_no(
                    bool(
                        not pit_frame.empty
                        and pit_frame["sample_fundamentals_used"].any()
                    )
                ),
            )

            if pit_frame.empty:
                st.warning("Not enough future history to validate these horizons.")
            else:
                accuracy_cols = st.columns(5)
                accuracy_cols[0].metric(
                    "Average error",
                    f"{accuracy['average_prediction_error_pct']:.1f}%",
                )
                accuracy_cols[1].metric(
                    "Median error",
                    f"{accuracy['median_prediction_error_pct']:.1f}%",
                )
                accuracy_cols[2].metric("RMSE", f"{accuracy['rmse_pct']:.1f}%")
                accuracy_cols[3].metric(
                    "Forecast/actual correlation",
                    f"{accuracy['forecast_actual_correlation']:.3f}",
                )
                accuracy_cols[4].metric(
                    "Calibration accuracy",
                    f"{accuracy['calibration_accuracy_pct']:.1f}%",
                )

                st.subheader("Score Threshold Validation")
                st.dataframe(threshold_frame, use_container_width=True)
                if not threshold_frame.empty:
                    st.bar_chart(
                        threshold_frame.pivot_table(
                            index="threshold",
                            columns="horizon",
                            values="average_actual_return_pct",
                        )
                    )

                st.subheader("Investment Outcome Validation")
                outcome_display = outcome_frame.copy()
                money_columns = [
                    column
                    for column in outcome_display.columns
                    if "value" in column or column == "total_invested"
                ]
                for column in money_columns:
                    outcome_display[column] = outcome_display[column].map(
                        lambda value: format_money(value, pit_currency)
                    )
                st.dataframe(outcome_display, use_container_width=True)

                st.subheader("Forecast Calibration")
                st.dataframe(calibration_frame, use_container_width=True)
                if not calibration_frame.empty:
                    calibration_horizon = st.selectbox(
                        "Point-in-time calibration horizon",
                        ["1 week", "1 month", "3 months", "6 months", "12 months", "3 years", "5 years"],
                    )
                    calibration_chart = (
                        calibration_frame[
                            calibration_frame["horizon"] == calibration_horizon
                        ]
                        .set_index("forecast_bucket")[
                            [
                                "average_forecast_return_pct",
                                "average_actual_return_pct",
                            ]
                        ]
                        .rename(
                            columns={
                                "average_forecast_return_pct": "Forecast",
                                "average_actual_return_pct": "Actual",
                            }
                        )
                    )
                    st.line_chart(calibration_chart)

                st.subheader("Point-in-Time Prediction Records")
                st.dataframe(pit_frame, use_container_width=True)
        st.caption(ADVICE_WARNING)


with tab_alpha_validation:
    alpha_universe_choice = st.selectbox(
        "Alpha validation universe",
        ["S&P 500", "Nasdaq 100", "Custom tickers"],
    )
    alpha_custom_universe = ""
    if alpha_universe_choice == "Custom tickers":
        alpha_custom_universe = st.text_area(
            "Alpha custom ticker universe",
            value="AAPL, MSFT, NVDA, AMZN, META, GOOGL, JPM, COST, TSLA, PLTR",
        )
    alpha_start = st.text_input("Alpha validation start date", value="2018-01-01")
    alpha_max_tickers = st.number_input(
        "Alpha maximum tickers",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )
    alpha_step_days = st.number_input(
        "Historical date step",
        min_value=1,
        max_value=252,
        value=21,
        step=1,
    )
    alpha_amount_cols = st.columns([2, 1])
    alpha_investment_amount = alpha_amount_cols[0].number_input(
        "Alpha investment amount",
        min_value=0.0,
        value=10.0,
        step=10.0,
    )
    alpha_currency = alpha_amount_cols[1].selectbox(
        "Alpha currency",
        ["$", "£", "€", "R"],
    )
    today_ticker = st.text_input("Today outcome ticker", value="META").upper().strip()

    if st.button("Run alpha validation"):
        if alpha_universe_choice == "S&P 500":
            alpha_tickers = get_universe("sp500")
        elif alpha_universe_choice == "Nasdaq 100":
            alpha_tickers = get_universe("nasdaq100")
        else:
            alpha_tickers = custom_tickers_from_text(alpha_custom_universe)

        alpha_tickers = alpha_tickers[: int(alpha_max_tickers)]
        if not alpha_tickers:
            st.warning("Add at least one ticker to run alpha validation.")
        else:
            price_history_by_ticker, fallback_flags = load_price_histories_for_tickers(
                alpha_tickers,
                start="2015-01-01",
            )
            if any(fallback_flags.values()):
                st.warning(YAHOO_WARNING)

            frame = alpha_validation_frame(
                price_history_by_ticker,
                start_date=alpha_start,
                fallback_data_used_by_ticker=fallback_flags,
                step_days=int(alpha_step_days),
            )
            bucket_frame = score_bucket_performance(frame)
            predictor_metrics = predictor_validation_metrics(frame)
            investment_frame = investment_outcome_by_bucket(
                bucket_frame,
                alpha_investment_amount,
            )
            threshold_frame = threshold_analysis(frame)

            spy_prices, _ = safe_price_history("SPY", start="2015-01-01")
            qqq_prices, _ = safe_price_history("QQQ", start="2015-01-01")
            benchmarks = {
                "S&P 500": benchmark_return_frame(
                    "S&P 500",
                    spy_prices,
                    start_date=alpha_start,
                    step_days=int(alpha_step_days),
                ),
                "Nasdaq 100": benchmark_return_frame(
                    "Nasdaq 100",
                    qqq_prices,
                    start_date=alpha_start,
                    step_days=int(alpha_step_days),
                ),
            }
            benchmark_frame = benchmark_comparison(frame, benchmarks)

            st.subheader("Validation Coverage")
            coverage_cols = st.columns(4)
            coverage_cols[0].metric("Tickers", len(alpha_tickers))
            coverage_cols[1].metric("Observations", len(frame))
            coverage_cols[2].metric(
                "Fallback tickers",
                sum(1 for value in fallback_flags.values() if value),
            )
            coverage_cols[3].metric(
                "Live tickers",
                sum(1 for value in fallback_flags.values() if not value),
            )

            if frame.empty:
                st.warning("Not enough history to build alpha validation.")
            else:
                st.subheader("Score Bucket Performance")
                st.dataframe(bucket_frame, use_container_width=True)

                st.subheader("Score Bucket vs Average Return")
                st.bar_chart(
                    bucket_frame.pivot(
                        index="score_bucket",
                        columns="horizon",
                        values="average_return_pct",
                    )
                )
                chart_cols = st.columns(3)
                chart_cols[0].write("Score Bucket vs Win Rate")
                chart_cols[0].bar_chart(
                    bucket_frame.pivot(
                        index="score_bucket",
                        columns="horizon",
                        values="win_rate_pct",
                    )
                )
                chart_cols[1].write("Score Bucket vs Sharpe")
                chart_cols[1].bar_chart(
                    bucket_frame.pivot(
                        index="score_bucket",
                        columns="horizon",
                        values="sharpe_ratio",
                    )
                )
                chart_cols[2].write("Score Bucket vs Drawdown")
                chart_cols[2].bar_chart(
                    bucket_frame.pivot(
                        index="score_bucket",
                        columns="horizon",
                        values="maximum_drawdown_pct",
                    )
                )

                st.subheader("Predictor Validation")
                metric_cols = st.columns(5)
                metric_cols[0].metric(
                    "Score/return correlation",
                    f"{predictor_metrics['correlation']:.3f}",
                )
                metric_cols[1].metric("R squared", f"{predictor_metrics['r_squared']:.3f}")
                metric_cols[2].metric(
                    "Information coefficient",
                    f"{predictor_metrics['information_coefficient']:.3f}",
                )
                metric_cols[3].metric(
                    "Hit rate",
                    f"{predictor_metrics['hit_rate_pct']:.1f}%",
                )
                metric_cols[4].metric(
                    "Calibration accuracy",
                    f"{predictor_metrics['calibration_accuracy_pct']:.1f}%",
                )
                st.metric("Predictive power", predictor_metrics["predictive_power"])

                st.subheader("Investment Simulation")
                simulation = investment_frame.copy()
                simulation["expected_value"] = simulation["expected_value"].map(
                    lambda value: format_money(value, alpha_currency)
                )
                simulation["profit_loss"] = simulation["profit_loss"].map(
                    lambda value: format_money(value, alpha_currency)
                )
                st.dataframe(simulation, use_container_width=True)

                st.subheader("Score Threshold Analysis")
                st.dataframe(threshold_frame, use_container_width=True)

                st.subheader("Benchmark Comparison")
                st.dataframe(benchmark_frame, use_container_width=True)

                today_prices, today_fallback = safe_price_history(today_ticker, period="5y")
                today_score, _ = score_for_app(today_ticker, today_prices, today_fallback)
                confidence = confidence_framework(
                    bucket_frame,
                    today_score.overall_score,
                    horizon="12 months",
                )
                st.subheader("Confidence Framework")
                confidence_cols = st.columns(5)
                confidence_cols[0].metric("Score", f"{today_score.overall_score:.2f}")
                confidence_cols[1].metric("Bucket", confidence["score_bucket"])
                confidence_cols[2].metric(
                    "12M positive probability",
                    f"{confidence['probability_positive_pct']:.1f}%",
                )
                confidence_cols[3].metric(
                    "12M probability of loss",
                    f"{confidence['probability_loss_pct']:.1f}%",
                )
                confidence_cols[4].metric("Confidence", confidence["confidence"])

                st.subheader("Final Investment Outcome Card")
                card_frame = final_investment_outcome_card(
                    bucket_frame,
                    today_score.overall_score,
                    alpha_investment_amount,
                )
                if card_frame.empty:
                    st.warning("No historical bucket observations for this score.")
                else:
                    card_display = card_frame.copy()
                    card_display["expected_value"] = card_display["expected_value"].map(
                        lambda value: format_money(value, alpha_currency)
                    )
                    st.write(
                        "If historical relationships continue, "
                        f"{format_money(alpha_investment_amount, alpha_currency)} "
                        "invested today is expected to become:"
                    )
                    st.dataframe(card_display, use_container_width=True)

                st.subheader("Stored Validation Records")
                st.dataframe(frame, use_container_width=True)
        st.caption(ADVICE_WARNING)


with tab_universe:
    universe_choice = st.selectbox(
        "Universe",
        ["S&P 500", "Nasdaq 100", "Custom tickers"],
    )
    custom_universe = ""
    if universe_choice == "Custom tickers":
        custom_universe = st.text_area(
            "Custom ticker universe",
            value="AAPL, MSFT, NVDA, AMZN, PLTR",
        )
    universe_start = st.text_input("Universe validation start date", value="2022-01-01")
    max_tickers = st.number_input(
        "Maximum tickers to test",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    if st.button("Run universe backtest"):
        if universe_choice == "S&P 500":
            universe_tickers = get_universe("sp500")
        elif universe_choice == "Nasdaq 100":
            universe_tickers = get_universe("nasdaq100")
        else:
            universe_tickers = custom_tickers_from_text(custom_universe)

        universe_tickers = universe_tickers[: int(max_tickers)]
        if not universe_tickers:
            st.warning("Add at least one ticker to run the universe backtest.")
        else:
            price_history_by_ticker = {}
            fallback_flags = {}
            progress = st.progress(0)
            for index, ticker in enumerate(universe_tickers, start=1):
                prices, is_fallback = safe_price_history(ticker, start="2018-01-01")
                price_history_by_ticker[ticker] = prices
                fallback_flags[ticker] = is_fallback
                progress.progress(index / len(universe_tickers))

            if any(fallback_flags.values()):
                st.warning(YAHOO_WARNING)

            frame = universe_prediction_frame(
                price_history_by_ticker,
                start_date=universe_start,
                fallback_data_used_by_ticker=fallback_flags,
            )
            summary = universe_bucket_summary(frame)
            calibration = universe_calibration_curve(frame)

            st.subheader("Universe Coverage")
            coverage_cols = st.columns(4)
            coverage_cols[0].metric("Tickers tested", len(universe_tickers))
            coverage_cols[1].metric("Observations", len(frame))
            coverage_cols[2].metric(
                "Fallback tickers",
                sum(1 for value in fallback_flags.values() if value),
            )
            coverage_cols[3].metric(
                "Live tickers",
                sum(1 for value in fallback_flags.values() if not value),
            )
            st.write(", ".join(universe_tickers))

            if frame.empty:
                st.warning("Not enough future price history for this universe/date.")
            else:
                st.subheader("Score Bucket Validation")
                st.dataframe(summary, use_container_width=True)

                average_return_chart = summary.pivot(
                    index="score_bucket",
                    columns="horizon",
                    values="average_return_pct",
                )
                st.subheader("Score Bucket vs Return")
                st.bar_chart(average_return_chart)

                win_rate_chart = summary.pivot(
                    index="score_bucket",
                    columns="horizon",
                    values="win_rate_pct",
                )
                st.subheader("Score Bucket vs Win Rate")
                st.bar_chart(win_rate_chart)

                st.subheader("Calibration Curve")
                st.dataframe(calibration, use_container_width=True)
                calibration_horizon = st.selectbox(
                    "Calibration horizon",
                    ["1 month", "3 months", "6 months", "12 months"],
                )
                calibration_chart = (
                    calibration[calibration["horizon"] == calibration_horizon]
                    .set_index("score_bucket")[
                        [
                            "average_predicted_return_pct",
                            "average_actual_return_pct",
                        ]
                    ]
                    .rename(
                        columns={
                            "average_predicted_return_pct": "Predicted return",
                            "average_actual_return_pct": "Actual return",
                        }
                    )
                )
                st.line_chart(calibration_chart)

                st.subheader("Prediction Observations")
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
