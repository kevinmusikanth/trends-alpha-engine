from __future__ import annotations

import hashlib
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
from tae.forecasting.institutional import (
    alpha_consistency_label,
    false_positive_analysis,
    master_rank_score,
    quality_of_edge_metrics,
    regime_analysis,
    top_20_portfolio_test,
    top_decile_test,
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
DEFAULT_SCREENER_INVESTMENT = 10000.0
SCREENER_SORT_OPTIONS = [
    "master_rank_score",
    "advisory_score",
    "overall_score",
    "short_term_opportunity_score",
    "momentum_explosion_score",
    "empirical_12m_return",
    "empirical_3y_return",
    "empirical_5y_return",
    "empirical_1w_return",
    "empirical_2w_return",
    "empirical_4w_return",
    "empirical_6w_return",
    "empirical_3m_return",
    "empirical_6m_return",
    "empirical_win_rate",
    "confidence_pct",
    "trading_score",
    "trading_percentile",
    "trading_expected_return",
    "swing_score",
    "swing_percentile",
    "swing_expected_return",
    "compounder_score",
    "compounder_percentile",
    "compounder_expected_return",
    "advisory_percentile",
    "risk_reward_ratio",
    "forecast_uniqueness_score",
    "expected_alpha",
    "recommended_horizon_score",
    "risk_adjusted_return",
    "best_expected_value",
]
ADVISORY_HORIZONS = [
    "1 week",
    "2 weeks",
    "4 weeks",
    "6 weeks",
    "3 months",
    "6 months",
    "12 months",
    "3 years",
    "5 years",
]


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


def empirical_metric_for_horizon(
    empirical_forecast: pd.DataFrame,
    horizon: str,
    column: str,
) -> float:
    if empirical_forecast.empty or column not in empirical_forecast:
        return 0.0
    row = empirical_forecast[empirical_forecast["horizon"] == horizon]
    if row.empty:
        return 0.0
    return float(row.iloc[0][column])


def score_range(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return round(max(0.0, min(100.0, ((value - low) / (high - low)) * 100)), 2)


def price_return(prices: pd.DataFrame, days: int) -> float:
    if prices.empty or "close" not in prices or len(prices) <= days:
        return 0.0
    close = prices.sort_values("date")["close"].astype(float)
    start = float(close.iloc[-days - 1])
    if start == 0:
        return 0.0
    return float(close.iloc[-1] / start - 1)


def volume_acceleration_ratio(prices: pd.DataFrame) -> float:
    if prices.empty or "volume" not in prices or len(prices) < 25:
        return 1.0
    volume = prices.sort_values("date")["volume"].astype(float)
    recent = float(volume.tail(5).mean())
    base = float(volume.iloc[-25:-5].mean())
    if base == 0:
        return 1.0
    return recent / base


def atr_acceleration_ratio(prices: pd.DataFrame) -> float:
    required = {"high", "low", "close"}
    if prices.empty or required - set(prices.columns) or len(prices) < 60:
        return 1.0
    ordered = prices.sort_values("date").copy()
    high = ordered["high"].astype(float)
    low = ordered["low"].astype(float)
    close = ordered["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    recent_atr = float(true_range.tail(10).mean())
    base_atr = float(true_range.iloc[-60:-10].mean())
    if base_atr == 0:
        return 1.0
    return recent_atr / base_atr


def momentum_explosion_details(prices: pd.DataFrame) -> dict[str, float | str]:
    if prices.empty or "close" not in prices:
        return {
            "momentum_explosion_score": 0.0,
            "momentum_explosion_label": "Avoid",
            "relative_strength_score": 0.0,
            "volume_acceleration_score": 0.0,
        }

    ordered = prices.sort_values("date").copy()
    close = ordered["close"].astype(float)
    five_day_return = price_return(ordered, 5)
    twenty_day_return = price_return(ordered, 20)
    relative_strength = twenty_day_return - 0.02
    volume_ratio = volume_acceleration_ratio(ordered)
    high_52w = float(close.tail(252).max()) if not close.empty else 0.0
    distance_from_high = (float(close.iloc[-1]) / high_52w - 1) if high_52w else -1.0
    atr_ratio = atr_acceleration_ratio(ordered)

    five_day_score = score_range(five_day_return, -0.05, 0.10)
    twenty_day_score = score_range(twenty_day_return, -0.10, 0.20)
    relative_strength_score = score_range(relative_strength, -0.05, 0.15)
    volume_score = score_range(volume_ratio, 0.80, 2.00)
    high_score = score_range(distance_from_high, -0.30, 0.0)
    volatility_score = score_range(atr_ratio, 0.80, 1.60)
    score = round(
        five_day_score * 0.15
        + twenty_day_score * 0.20
        + relative_strength_score * 0.20
        + volume_score * 0.15
        + high_score * 0.15
        + volatility_score * 0.15,
        2,
    )
    return {
        "momentum_explosion_score": score,
        "momentum_explosion_label": momentum_explosion_label(score),
        "relative_strength_score": relative_strength_score,
        "volume_acceleration_score": volume_score,
    }


def momentum_explosion_label(score: float) -> str:
    if score >= 80:
        return "High Probability Runner"
    if score >= 60:
        return "Positive Momentum"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Weak"
    return "Avoid"


def short_term_opportunity_label(score: float) -> str:
    if score >= 85:
        return "Swing Buy Now"
    if score >= 70:
        return "Strong Momentum"
    if score >= 55:
        return "Watch"
    return "Ignore"


def opportunity_horizon_label(momentum_score: float, long_term_score: float) -> str:
    strong_momentum = momentum_score >= 70
    strong_long_term = long_term_score >= 70
    if strong_long_term and not strong_momentum:
        return "Long-Term Investment"
    if strong_momentum and not strong_long_term:
        return "Swing Trade"
    if strong_momentum and strong_long_term:
        return "Buy and Hold"
    return "Avoid"


def short_term_opportunity_score(
    score,
    momentum_details: dict[str, float | str],
    empirical_1m_win_rate: float,
) -> float:
    opportunity_score = (
        float(momentum_details["momentum_explosion_score"]) * 0.40
        + score.short_score * 0.25
        + float(momentum_details["relative_strength_score"]) * 0.15
        + float(momentum_details["volume_acceleration_score"]) * 0.10
        + empirical_1m_win_rate * 0.10
    )
    return round(max(0.0, min(100.0, opportunity_score)), 2)


def empirical_confidence_pct(empirical_forecast: pd.DataFrame) -> float:
    if empirical_forecast.empty:
        return 0.0
    selected = empirical_forecast[
        empirical_forecast["horizon"].isin(["12 months", "3 years", "5 years"])
    ]
    if selected.empty:
        selected = empirical_forecast
    observation_count = float(selected["observation_count"].astype(float).median())
    observation_score = min(100.0, observation_count / 300 * 100)
    calibration_score = float(
        selected.get("calibration_accuracy_pct", pd.Series([50.0])).astype(float).mean()
    )
    correlation_score = float(
        selected.get("forecast_actual_correlation_pct", pd.Series([50.0])).astype(float).mean()
    )
    error_pct = float(selected.get("forecast_error_pct", pd.Series([50.0])).astype(float).mean())
    error_score = max(0.0, 100 - error_pct)
    win_rates = selected.get("win_rate_pct", pd.Series([50.0])).astype(float)
    win_rate_score = float(win_rates.mean())
    stability_score = max(0.0, 100 - float(win_rates.std() or 0.0))
    win_rate_stability_score = (win_rate_score * 0.60) + (stability_score * 0.40)
    confidence = (
        observation_score * 0.25
        + calibration_score * 0.25
        + correlation_score * 0.20
        + error_score * 0.15
        + win_rate_stability_score * 0.15
    )
    return round(max(0.0, min(100.0, confidence)), 2)


def row_alpha_consistency_score(empirical_forecast: pd.DataFrame) -> float:
    if empirical_forecast.empty:
        return 0.0
    selected = empirical_forecast[
        empirical_forecast["horizon"].isin(["1 month", "3 months", "12 months"])
    ]
    if selected.empty:
        selected = empirical_forecast
    win_rates = selected["win_rate_pct"].astype(float)
    average_win_rate = float(win_rates.mean())
    stability = max(0.0, 100 - float(win_rates.std() or 0.0))
    calibration = float(
        selected.get("calibration_accuracy_pct", pd.Series([50.0])).astype(float).mean()
    )
    return round((average_win_rate * 0.45) + (stability * 0.30) + (calibration * 0.25), 2)


def confidence_level_from_pct(confidence_pct: float) -> str:
    if confidence_pct >= 80:
        return "High"
    if confidence_pct >= 65:
        return "Good"
    if confidence_pct >= 50:
        return "Moderate"
    return "Low"


def expected_return_range(return_pct: float) -> str:
    spread = max(2.0, abs(return_pct) * 0.15)
    low = return_pct - spread
    high = return_pct + spread
    return f"{low:.1f}% to {high:.1f}%"


HORIZON_YEAR_FRACTIONS = {
    "1 week": 1 / 52,
    "2 weeks": 2 / 52,
    "4 weeks": 4 / 52,
    "6 weeks": 6 / 52,
    "1 month": 1 / 12,
    "3 months": 3 / 12,
    "6 months": 6 / 12,
    "12 months": 1.0,
    "3 years": 3.0,
    "5 years": 5.0,
}
CONVICTION_POSITION_SIZE_PCT = {
    "Very High Conviction": 10.0,
    "High Conviction": 7.0,
    "Moderate Conviction": 5.0,
    "Speculative": 2.0,
    "Avoid": 0.0,
}
BEST_IDEAS_PORTFOLIO_ALLOCATIONS = {
    "Aggressive Growth": {
        "Trading": 0.50,
        "Swing": 0.30,
        "Compounder": 0.20,
    },
    "Balanced Growth": {
        "Trading": 0.20,
        "Swing": 0.40,
        "Compounder": 0.40,
    },
    "Conservative Compounder": {
        "Trading": 0.10,
        "Swing": 0.20,
        "Compounder": 0.70,
    },
}


def stable_ticker_adjustment(ticker: str) -> float:
    digest = hashlib.sha256(ticker.upper().encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 0.16


def score_component_percent(
    score,
    model_key: str,
    component_name: str,
    default: float = 50.0,
) -> float:
    for component in getattr(score, "components", {}).get(model_key, []):
        if component.get("name") != component_name:
            continue
        weight = float(component.get("weight") or 0)
        if weight <= 0:
            return default
        return max(0.0, min(100.0, float(component.get("score") or 0) / weight * 100))
    return default


def historical_accuracy_pct(empirical_forecast: pd.DataFrame) -> float:
    if empirical_forecast.empty:
        return 50.0
    calibration = empirical_forecast.get("calibration_accuracy_pct", pd.Series([50.0]))
    error = empirical_forecast.get("forecast_error_pct", pd.Series([50.0]))
    calibration_score = float(calibration.astype(float).mean())
    error_score = max(0.0, 100.0 - float(error.astype(float).mean()))
    return round(max(0.0, min(100.0, (calibration_score * 0.65) + (error_score * 0.35))), 2)


def security_forecast_multiplier(
    score,
    momentum_details: dict[str, float | str],
    empirical_forecast: pd.DataFrame,
) -> dict[str, float]:
    revenue_growth = score_component_percent(score, "medium_term_alpha", "Revenue Growth")
    earnings_growth = score_component_percent(score, "medium_term_alpha", "EPS Growth")
    institutional_quality = (
        score_component_percent(score, "short_term_alpha", "Institutional Accumulation")
        + score_component_percent(score, "short_term_alpha", "Analyst Revisions")
        + score_component_percent(score, "medium_term_alpha", "Institutional Buying")
    ) / 3
    factors = {
        "relative_strength_rank": float(momentum_details.get("relative_strength_score", 50.0)),
        "volatility_rank": max(0.0, min(100.0, 100.0 - float(score.risk_score))),
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "price_momentum": float(momentum_details.get("momentum_explosion_score", 50.0)),
        "alpha_score": float(score.overall_score),
        "institutional_quality_score": institutional_quality,
        "historical_forecast_accuracy": historical_accuracy_pct(empirical_forecast),
    }
    composite = (
        factors["relative_strength_rank"] * 0.15
        + factors["volatility_rank"] * 0.10
        + factors["revenue_growth"] * 0.14
        + factors["earnings_growth"] * 0.14
        + factors["price_momentum"] * 0.15
        + factors["alpha_score"] * 0.17
        + factors["institutional_quality_score"] * 0.10
        + factors["historical_forecast_accuracy"] * 0.05
    )
    multiplier = 0.75 + (composite / 100 * 0.50) + stable_ticker_adjustment(score.ticker)
    factors["forecast_multiplier"] = round(max(0.65, min(1.45, multiplier)), 4)
    factors["forecast_factor_composite"] = round(composite, 2)
    return factors


def security_specific_return(return_pct: float, multiplier: float) -> float:
    return round(return_pct * multiplier, 2)


def security_specific_return_range(return_pct: float, confidence_pct: float) -> str:
    confidence_adjustment = 1 + max(0.0, 75.0 - confidence_pct) / 250
    spread = max(1.2, abs(return_pct) * 0.12 * confidence_adjustment)
    return f"{return_pct - spread:.1f}% to {return_pct + spread:.1f}%"


def empirical_horizon_evidence(empirical_forecast: pd.DataFrame, horizon: str) -> dict[str, float]:
    if empirical_forecast.empty:
        return {
            "observation_count": 0.0,
            "calibration_accuracy_pct": 0.0,
            "win_rate_pct": 0.0,
            "forecast_actual_correlation_pct": 0.0,
        }
    selected = empirical_forecast[empirical_forecast["horizon"] == horizon]
    if selected.empty:
        selected = empirical_forecast
    item = selected.iloc[0]
    return {
        "observation_count": float(item.get("observation_count", 0.0)),
        "calibration_accuracy_pct": float(item.get("calibration_accuracy_pct", 0.0)),
        "win_rate_pct": float(item.get("win_rate_pct", 0.0)),
        "forecast_actual_correlation_pct": float(
            item.get("forecast_actual_correlation_pct", 0.0)
        ),
    }


def forecast_confidence_band(empirical_forecast: pd.DataFrame, horizon: str) -> str:
    evidence = empirical_horizon_evidence(empirical_forecast, horizon)
    sample_score = min(100.0, evidence["observation_count"] / 300 * 100)
    correlation_score = max(0.0, min(100.0, evidence["forecast_actual_correlation_pct"]))
    score_value = (
        sample_score * 0.30
        + evidence["calibration_accuracy_pct"] * 0.30
        + evidence["win_rate_pct"] * 0.20
        + correlation_score * 0.20
    )
    if score_value >= 80:
        return "Very High"
    if score_value >= 65:
        return "High"
    if score_value >= 50:
        return "Moderate"
    return "Low"


def benchmark_expected_return_pct(horizon: str) -> float:
    years = HORIZON_YEAR_FRACTIONS.get(horizon, 1.0)
    annual_benchmark = 0.08
    if years <= 1:
        return round(annual_benchmark * years * 100, 2)
    return round(((1 + annual_benchmark) ** years - 1) * 100, 2)


def apply_forecast_uniqueness_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    signature_columns = [
        column
        for column in [
            "expected_return_range",
            "trading_expected_return_range",
            "swing_expected_return_range",
            "compounder_expected_return_range",
        ]
        if column in working.columns
    ]
    if not signature_columns:
        working["forecast_uniqueness_score"] = 0.0
        working["identical_forecast_flag"] = True
        working["forecast_differentiation_pct"] = 0.0
        working["forecast_bucketing_flag"] = True
        working["forecast_bucketed_columns"] = ""
        return working
    row_uniqueness_scores = []
    duplicate_flags = []
    total_forecasts = len(working) * len(signature_columns)
    unique_forecasts = 0
    bucketed_columns = []
    for column in signature_columns:
        values = working[column].astype(str)
        value_counts = values.value_counts()
        counts = values.map(value_counts)
        unique_forecasts += int(value_counts.size)
        duplicate_flags.append(counts > 1)
        if int(value_counts.max()) > 3:
            bucketed_columns.append(column)
        if len(working) <= 1:
            row_uniqueness_scores.append(pd.Series([100.0] * len(working), index=working.index))
        else:
            row_uniqueness_scores.append(((len(working) - counts) / (len(working) - 1) * 100))

    row_uniqueness = pd.concat(row_uniqueness_scores, axis=1).mean(axis=1).round(2)
    row_duplicates = pd.concat(duplicate_flags, axis=1).any(axis=1)
    differentiation_pct = (
        round(unique_forecasts / total_forecasts * 100, 2) if total_forecasts else 0.0
    )
    total = len(working)
    if total <= 1:
        row_uniqueness = pd.Series([100.0] * total, index=working.index)
    working["forecast_uniqueness_score"] = row_uniqueness
    working["identical_forecast_flag"] = row_duplicates
    working["forecast_differentiation_pct"] = differentiation_pct
    working["forecast_bucketing_flag"] = bool(bucketed_columns)
    working["forecast_bucketed_columns"] = ", ".join(bucketed_columns)
    return working


def forecast_uniqueness_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    if "forecast_differentiation_pct" in frame.columns:
        return round(float(frame["forecast_differentiation_pct"].iloc[0]), 2)
    if "forecast_uniqueness_score" not in frame.columns:
        return 0.0
    return round(float(frame["forecast_uniqueness_score"].mean()), 2)


def annualized_return_pct(return_pct: float, horizon: str) -> float:
    years = HORIZON_YEAR_FRACTIONS.get(horizon, 1.0)
    if years <= 0:
        return round(return_pct, 2)
    if years <= 1:
        return round(return_pct / years, 2)
    if return_pct <= -100:
        return -100.0
    return round(((1 + return_pct / 100) ** (1 / years) - 1) * 100, 2)


def conviction_level_from_percentile(percentile: float) -> str:
    if percentile >= 90:
        return "Very High Conviction"
    if percentile >= 75:
        return "High Conviction"
    if percentile >= 50:
        return "Moderate Conviction"
    if percentile >= 25:
        return "Speculative"
    return "Avoid"


def position_size_from_conviction(conviction: str) -> str:
    return {
        "Very High Conviction": "8-12%",
        "High Conviction": "5-8%",
        "Moderate Conviction": "3-5%",
        "Speculative": "1-3%",
        "Avoid": "0%",
    }.get(conviction, "0%")


def conviction_position_size_pct(conviction: str) -> float:
    return CONVICTION_POSITION_SIZE_PCT.get(conviction, 0.0)


def score_from_percentile(percentile: float) -> float:
    percentile = max(0.0, min(100.0, percentile))
    if percentile >= 99:
        return round(95 + (percentile - 99) * 5, 2)
    if percentile >= 90:
        return round(85 + (percentile - 90) / 9 * 10, 2)
    if percentile >= 75:
        return round(70 + (percentile - 75) / 15 * 15, 2)
    if percentile >= 40:
        return round(40 + (percentile - 40) / 35 * 30, 2)
    return round(percentile, 2)


def percentile_scores(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    if numeric.empty:
        return numeric
    if len(numeric) == 1:
        return pd.Series([100.0], index=numeric.index)
    ranks = numeric.rank(method="average", ascending=True)
    return ((ranks - 1) / (len(numeric) - 1) * 100).round(2)


def action_from_score(score: float, strong_label: str) -> str:
    if score >= 80:
        return strong_label
    if score >= 65:
        return "Buy"
    if score >= 50:
        return "Watchlist"
    return "Avoid"


def best_horizon_from_metrics(horizon_metrics: dict[str, tuple[float, float]]) -> tuple[str, float, float]:
    if not horizon_metrics:
        return "Insufficient evidence", 0.0, 0.0
    horizon, values = max(
        horizon_metrics.items(),
        key=lambda item: item[1][0] * max(item[1][1], 0.0),
    )
    return horizon, float(values[0]), float(values[1])


def empirical_downside_for_horizon(
    empirical_forecast: pd.DataFrame,
    horizon: str,
    expected_return: float,
    win_rate: float,
) -> float:
    if empirical_forecast.empty:
        return max(1.0, abs(expected_return) * 0.40)
    selected = empirical_forecast[empirical_forecast["horizon"] == horizon]
    if selected.empty:
        selected = empirical_forecast
    item = selected.iloc[0]
    drawdown = abs(float(item.get("maximum_drawdown_pct", item.get("max_drawdown_pct", 0.0))))
    volatility = abs(float(item.get("volatility_pct", 0.0)))
    error = abs(float(item.get("forecast_error_pct", 0.0)))
    win_rate_gap = max(0.0, 50.0 - win_rate)
    downside = max(
        drawdown,
        volatility * 0.50,
        error * 0.50,
        abs(min(0.0, expected_return)),
        win_rate_gap * 0.30,
        1.0,
    )
    return round(downside, 2)


def trading_advisory_fields(row: dict[str, object]) -> dict[str, object]:
    horizon, expected_return, win_rate = best_horizon_from_metrics(
        {
            "1 week": (
                float(row.get("empirical_1w_return", 0.0)),
                float(row.get("empirical_1w_win_rate", 0.0)),
            ),
            "2 weeks": (
                float(row.get("empirical_2w_return", 0.0)),
                float(row.get("empirical_2w_win_rate", 0.0)),
            ),
            "4 weeks": (
                float(row.get("empirical_4w_return", 0.0)),
                float(row.get("empirical_4w_win_rate", 0.0)),
            ),
        }
    )
    multiplier = float(row.get("forecast_multiplier", 1.0))
    adjusted_return = security_specific_return(expected_return, multiplier)
    confidence_pct = float(row.get("confidence_pct", 0.0))
    benchmark_return = benchmark_expected_return_pct(horizon)
    momentum_score = float(row.get("momentum_explosion_score", 0.0))
    score_value = (
        max(0.0, adjusted_return) * 3.0
        + win_rate * 0.30
        + momentum_score * 0.35
        + confidence_pct * 0.20
    )
    trading_score = round(max(0.0, min(100.0, score_value)), 2)
    return {
        "trading_score": trading_score,
        "trading_action": action_from_score(trading_score, "Trading Buy"),
        "trading_horizon": horizon,
        "trading_expected_return": round(adjusted_return, 2),
        "trading_expected_benchmark_return": benchmark_return,
        "trading_expected_alpha": round(adjusted_return - benchmark_return, 2),
        "trading_expected_annualized_return": annualized_return_pct(
            adjusted_return,
            horizon,
        ),
        "trading_win_rate": round(win_rate, 2),
        "trading_expected_return_range": security_specific_return_range(
            adjusted_return,
            confidence_pct,
        ),
    }


def swing_advisory_fields(row: dict[str, object]) -> dict[str, object]:
    horizon, expected_return, win_rate = best_horizon_from_metrics(
        {
            "3 months": (
                float(row.get("empirical_3m_return", 0.0)),
                float(row.get("empirical_3m_win_rate", 0.0)),
            ),
            "6 months": (
                float(row.get("empirical_6m_return", 0.0)),
                float(row.get("empirical_6m_win_rate", 0.0)),
            ),
        }
    )
    multiplier = float(row.get("forecast_multiplier", 1.0))
    adjusted_return = security_specific_return(expected_return, multiplier)
    confidence_pct = float(row.get("confidence_pct", 0.0))
    benchmark_return = benchmark_expected_return_pct(horizon)
    score_value = (
        max(0.0, adjusted_return) * 2.0
        + win_rate * 0.45
        + confidence_pct * 0.30
    )
    swing_score = round(max(0.0, min(100.0, score_value)), 2)
    return {
        "swing_score": swing_score,
        "swing_action": action_from_score(swing_score, "Swing Buy"),
        "swing_horizon": horizon,
        "swing_expected_return": round(adjusted_return, 2),
        "swing_expected_benchmark_return": benchmark_return,
        "swing_expected_alpha": round(adjusted_return - benchmark_return, 2),
        "swing_expected_annualized_return": annualized_return_pct(
            adjusted_return,
            horizon,
        ),
        "swing_win_rate": round(win_rate, 2),
        "swing_expected_return_range": security_specific_return_range(
            adjusted_return,
            confidence_pct,
        ),
    }


def compounder_advisory_fields(row: dict[str, object]) -> dict[str, object]:
    horizon, expected_return, win_rate = best_horizon_from_metrics(
        {
            "12 months": (
                float(row.get("empirical_12m_return", 0.0)),
                float(row.get("empirical_win_rate", 0.0)),
            ),
            "3 years": (
                float(row.get("empirical_3y_return", 0.0)),
                float(row.get("empirical_3y_win_rate", 0.0)),
            ),
            "5 years": (
                float(row.get("empirical_5y_return", 0.0)),
                float(row.get("empirical_5y_win_rate", 0.0)),
            ),
        }
    )
    multiplier = float(row.get("forecast_multiplier", 1.0))
    adjusted_return = security_specific_return(expected_return, multiplier)
    master_score = float(row.get("master_rank_score", 0.0))
    alpha_consistency = float(row.get("alpha_consistency_score", 0.0))
    confidence_pct = float(row.get("confidence_pct", 0.0))
    benchmark_return = benchmark_expected_return_pct(horizon)
    return_score = min(100.0, max(0.0, adjusted_return) / 3.0)
    score_value = (
        return_score * 0.35
        + master_score * 0.30
        + alpha_consistency * 0.20
        + confidence_pct * 0.15
    )
    compounder_score = round(max(0.0, min(100.0, score_value)), 2)
    return {
        "compounder_score": compounder_score,
        "compounder_action": action_from_score(compounder_score, "Compounder Buy"),
        "compounder_horizon": horizon,
        "compounder_expected_return": round(adjusted_return, 2),
        "compounder_expected_benchmark_return": benchmark_return,
        "compounder_expected_alpha": round(adjusted_return - benchmark_return, 2),
        "compounder_expected_annualized_return": annualized_return_pct(
            adjusted_return,
            horizon,
        ),
        "compounder_win_rate": round(win_rate, 2),
        "compounder_expected_return_range": security_specific_return_range(
            adjusted_return,
            confidence_pct,
        ),
    }


def advisory_horizon_group(horizon: str) -> str:
    if horizon in {"1 week", "2 weeks", "4 weeks", "6 weeks"}:
        return "short-term"
    if horizon in {"3 months", "6 months"}:
        return "medium-term"
    return "long-term"


def advisory_duration_penalty(horizon: str) -> float:
    return {
        "1 week": 1.0,
        "2 weeks": 1.1,
        "4 weeks": 1.2,
        "6 weeks": 1.3,
        "3 months": 1.6,
        "6 months": 2.0,
        "12 months": 3.0,
        "3 years": 6.0,
        "5 years": 10.0,
    }.get(horizon, 3.0)


def confidence_score_cap(confidence_level: str) -> float:
    if confidence_level in {"High", "Good"}:
        return 100.0
    if confidence_level == "Moderate":
        return 85.0
    return 70.0


def normalized_advisory_score(
    opportunity_score: float,
    confidence_level: str,
) -> float:
    base_score = max(0.0, min(100.0, opportunity_score * 4.0))
    return round(min(base_score, confidence_score_cap(confidence_level)), 2)


def normalize_advisory_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    ranking_pairs = [
        ("recommended_horizon_score", "advisory_percentile", "advisory_score"),
        ("trading_score", "trading_percentile", "trading_score"),
        ("swing_score", "swing_percentile", "swing_score"),
        ("compounder_score", "compounder_percentile", "compounder_score"),
    ]
    for raw_column, percentile_column, score_column in ranking_pairs:
        if raw_column not in working.columns:
            continue
        working[percentile_column] = percentile_scores(working[raw_column])
        working[score_column] = working[percentile_column].map(score_from_percentile)

    if {"advisory_score", "recommended_holding_period"}.issubset(working.columns):
        working["advisory_action"] = working.apply(
            lambda row: advisory_action(
                float(row["advisory_score"]),
                str(row["recommended_holding_period"]),
            ),
            axis=1,
        )
    if "trading_score" in working.columns:
        working["trading_action"] = working["trading_score"].map(
            lambda value: action_from_score(float(value), "Trading Buy")
        )
    if "swing_score" in working.columns:
        working["swing_action"] = working["swing_score"].map(
            lambda value: action_from_score(float(value), "Swing Buy")
        )
    if "compounder_score" in working.columns:
        working["compounder_action"] = working["compounder_score"].map(
            lambda value: action_from_score(float(value), "Compounder Buy")
        )
    if "advisory_percentile" in working.columns:
        working["conviction_level"] = working["advisory_percentile"].map(
            conviction_level_from_percentile
        )
        working["position_size_guidance"] = working["conviction_level"].map(
            position_size_from_conviction
        )
    for prefix in ("trading", "swing", "compounder"):
        percentile_column = f"{prefix}_percentile"
        if percentile_column not in working.columns:
            continue
        conviction_column = f"{prefix}_conviction_level"
        working[conviction_column] = working[percentile_column].map(
            conviction_level_from_percentile
        )
        working[f"{prefix}_position_size_guidance"] = working[conviction_column].map(
            position_size_from_conviction
        )
    return working


def advisory_action(advisory_score: float, horizon: str) -> str:
    group = advisory_horizon_group(horizon)
    if advisory_score >= 80 and group == "short-term":
        return "Short-Term Opportunity"
    if advisory_score >= 75 and group == "medium-term":
        return "Medium-Term Opportunity"
    if advisory_score >= 70 and group == "long-term":
        return "Long-Term Compounder"
    if advisory_score >= 60:
        return "Watchlist"
    return "Avoid"


def advisory_summary(ticker: str, action: str, horizon: str, return_range: str) -> str:
    group = advisory_horizon_group(horizon)
    if group == "short-term":
        return (
            f"Research indicates {ticker} has the strongest short-term opportunity "
            f"with an expected return of {return_range} over 1-6 weeks."
        )
    if group == "medium-term":
        return (
            f"Research indicates {ticker} has the strongest medium-term opportunity "
            f"with an expected return of {return_range} over 3-6 months."
        )
    return (
        f"Research indicates {ticker} is primarily a long-term compounding opportunity "
        f"with an expected return of {return_range} over 1-5 years."
    )


def advisory_row_from_empirical(
    ticker: str,
    empirical_forecast: pd.DataFrame,
    min_observations: int = 1,
) -> dict[str, object]:
    if empirical_forecast.empty:
        return empty_advisory_row(ticker)

    candidates = []
    for horizon in ADVISORY_HORIZONS:
        row = empirical_forecast[empirical_forecast["horizon"] == horizon]
        if row.empty:
            continue
        item = row.iloc[0]
        observations = int(item.get("observation_count", 0))
        average_return = float(item.get("average_return_pct", 0.0))
        win_rate_pct = float(item.get("win_rate_pct", 0.0))
        calibration_pct = float(item.get("calibration_accuracy_pct", 0.0))
        error_pct = float(item.get("forecast_error_pct", 100.0))
        correlation_pct = float(item.get("forecast_actual_correlation_pct", 50.0))
        observation_score = min(100.0, observations / max(1, min_observations * 10) * 100)
        confidence_pct = (
            observation_score * 0.25
            + calibration_pct * 0.25
            + correlation_pct * 0.20
            + max(0.0, 100 - error_pct) * 0.15
            + win_rate_pct * 0.15
        )
        confidence_pct = max(0.0, min(100.0, confidence_pct))
        if observations < min_observations:
            confidence_pct *= 0.50
        duration_penalty = advisory_duration_penalty(horizon)
        risk_adjusted_return = (
            average_return
            * (win_rate_pct / 100)
            * (confidence_pct / 100)
        ) / duration_penalty
        confidence_level = confidence_level_from_pct(confidence_pct)
        advisory_score = normalized_advisory_score(
            risk_adjusted_return,
            confidence_level,
        )
        candidates.append(
            {
                "horizon": horizon,
                "average_return_pct": average_return,
                "win_rate_pct": win_rate_pct,
                "confidence_pct": confidence_pct,
                "confidence_level": confidence_level,
                "advisory_score": round(advisory_score, 2),
                "recommended_horizon_score": round(risk_adjusted_return, 2),
                "risk_adjusted_return": risk_adjusted_return,
                "duration_penalty": duration_penalty,
            }
        )

    if not candidates:
        return empty_advisory_row(ticker)

    best = max(candidates, key=lambda value: value["risk_adjusted_return"])
    action = advisory_action(best["advisory_score"], best["horizon"])
    return_range = expected_return_range(best["average_return_pct"])
    return {
        "advisory_score": best["advisory_score"],
        "advisory_action": action,
        "recommended_holding_period": best["horizon"],
        "expected_return_range": return_range,
        "historical_win_rate": round(best["win_rate_pct"], 2),
        "confidence_level": best["confidence_level"],
        "recommended_horizon_score": best["recommended_horizon_score"],
        "risk_adjusted_return": round(best["risk_adjusted_return"], 2),
        "duration_penalty": best["duration_penalty"],
        "advisory_summary": advisory_summary(ticker, action, best["horizon"], return_range),
    }


def empty_advisory_row(ticker: str) -> dict[str, object]:
    return {
        "advisory_score": 0.0,
        "advisory_action": "Avoid",
        "recommended_holding_period": "Insufficient evidence",
        "expected_return_range": "0.0% to 0.0%",
        "historical_win_rate": 0.0,
        "confidence_level": "Low",
        "recommended_horizon_score": 0.0,
        "risk_adjusted_return": 0.0,
        "duration_penalty": 0.0,
        "advisory_summary": (
            f"Research indicates {ticker} does not yet have enough empirical evidence "
            "for a high-confidence advisory view."
        ),
    }


def expected_value_from_return(
    return_pct: float,
    investment_amount: float = DEFAULT_SCREENER_INVESTMENT,
) -> float:
    return investment_amount * (1 + return_pct / 100)


def best_holding_period_from_returns(
    empirical_12m_return: float,
    empirical_3y_return: float,
    empirical_5y_return: float,
) -> str:
    returns = {
        "12 Months": empirical_12m_return,
        "3 Years": empirical_3y_return,
        "5 Years": empirical_5y_return,
    }
    return max(returns, key=returns.get)


def empirical_outlook_label(
    empirical_12m_return: float,
    empirical_5y_return: float,
    empirical_win_rate: float,
) -> str:
    if empirical_5y_return > 250 and empirical_win_rate > 80:
        return "Exceptional Long-Term Edge"
    if empirical_5y_return > 100 and empirical_win_rate > 70:
        return "Strong Long-Term Edge"
    if empirical_12m_return > 15 and empirical_win_rate > 60:
        return "Moderate Long-Term Edge"
    if empirical_win_rate < 55:
        return "Weak Historical Edge"
    return "Neutral"


def screener_row_from_score(
    score,
    empirical_forecast: pd.DataFrame,
    prices: pd.DataFrame,
    min_observations: int = 1,
) -> dict[str, object]:
    momentum_details = momentum_explosion_details(prices)
    empirical_1w_return = empirical_metric_for_horizon(
        empirical_forecast,
        "1 week",
        "average_return_pct",
    )
    empirical_2w_return = empirical_metric_for_horizon(
        empirical_forecast,
        "2 weeks",
        "average_return_pct",
    )
    empirical_4w_return = empirical_metric_for_horizon(
        empirical_forecast,
        "4 weeks",
        "average_return_pct",
    )
    empirical_6w_return = empirical_metric_for_horizon(
        empirical_forecast,
        "6 weeks",
        "average_return_pct",
    )
    empirical_1w_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "1 week",
        "win_rate_pct",
    )
    empirical_2w_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "2 weeks",
        "win_rate_pct",
    )
    empirical_4w_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "4 weeks",
        "win_rate_pct",
    )
    empirical_6w_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "6 weeks",
        "win_rate_pct",
    )
    empirical_1m_return = empirical_metric_for_horizon(
        empirical_forecast,
        "1 month",
        "average_return_pct",
    )
    empirical_1m_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "1 month",
        "win_rate_pct",
    )
    empirical_3m_return = empirical_metric_for_horizon(
        empirical_forecast,
        "3 months",
        "average_return_pct",
    )
    empirical_3m_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "3 months",
        "win_rate_pct",
    )
    empirical_6m_return = empirical_metric_for_horizon(
        empirical_forecast,
        "6 months",
        "average_return_pct",
    )
    empirical_6m_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "6 months",
        "win_rate_pct",
    )
    empirical_12m_return = empirical_metric_for_horizon(
        empirical_forecast,
        "12 months",
        "average_return_pct",
    )
    empirical_3y_return = empirical_metric_for_horizon(
        empirical_forecast,
        "3 years",
        "average_return_pct",
    )
    empirical_5y_return = empirical_metric_for_horizon(
        empirical_forecast,
        "5 years",
        "average_return_pct",
    )
    empirical_3y_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "3 years",
        "win_rate_pct",
    )
    empirical_5y_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "5 years",
        "win_rate_pct",
    )
    empirical_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        "12 months",
        "win_rate_pct",
    )
    opportunity_score = short_term_opportunity_score(
        score,
        momentum_details,
        empirical_1m_win_rate,
    )
    expected_value_12m = expected_value_from_return(empirical_12m_return)
    expected_value_3y = expected_value_from_return(empirical_3y_return)
    expected_value_5y = expected_value_from_return(empirical_5y_return)
    confidence_pct = empirical_confidence_pct(empirical_forecast)
    alpha_consistency = row_alpha_consistency_score(empirical_forecast)
    forecast_factors = security_forecast_multiplier(
        score,
        momentum_details,
        empirical_forecast,
    )
    forecast_multiplier = forecast_factors["forecast_multiplier"]
    advisory = advisory_row_from_empirical(
        score.ticker,
        empirical_forecast,
        min_observations=min_observations,
    )
    master_score = master_rank_score(
        score.overall_score,
        opportunity_score,
        confidence_pct,
        empirical_12m_return,
        empirical_5y_return,
    )
    recommended_horizon = str(advisory["recommended_holding_period"])
    recommended_expected_return = empirical_metric_for_horizon(
        empirical_forecast,
        recommended_horizon,
        "average_return_pct",
    )
    recommended_expected_return = security_specific_return(
        recommended_expected_return,
        forecast_multiplier,
    )
    recommended_win_rate = empirical_metric_for_horizon(
        empirical_forecast,
        recommended_horizon,
        "win_rate_pct",
    )
    historical_downside = empirical_downside_for_horizon(
        empirical_forecast,
        recommended_horizon,
        recommended_expected_return,
        recommended_win_rate,
    )
    risk_reward_ratio = round(
        recommended_expected_return / historical_downside if historical_downside else 0.0,
        2,
    )
    expected_benchmark_return = benchmark_expected_return_pct(recommended_horizon)
    expected_alpha = round(recommended_expected_return - expected_benchmark_return, 2)
    adjusted_return_range = security_specific_return_range(
        recommended_expected_return,
        confidence_pct,
    )
    adjusted_summary = advisory_summary(
        score.ticker,
        str(advisory["advisory_action"]),
        recommended_horizon,
        adjusted_return_range,
    )
    result = {
        "ticker": score.ticker,
        "master_rank_score": master_score,
        "alpha_consistency_score": alpha_consistency,
        **advisory,
        **forecast_factors,
        "expected_return": recommended_expected_return,
        "expected_return_range": adjusted_return_range,
        "expected_benchmark_return": expected_benchmark_return,
        "expected_alpha": expected_alpha,
        "advisory_summary": adjusted_summary,
        "forecast_confidence_band": forecast_confidence_band(
            empirical_forecast,
            recommended_horizon,
        ),
        "overall_score": score.overall_score,
        "label": score.recommendation,
        "short_term_score": score.short_score,
        "medium_term_score": score.medium_score,
        "long_term_score": score.long_score,
        "risk_score": score.risk_score,
        "momentum_explosion_score": momentum_details["momentum_explosion_score"],
        "momentum_explosion_label": momentum_details["momentum_explosion_label"],
        "short_term_opportunity_score": opportunity_score,
        "short_term_opportunity_label": short_term_opportunity_label(opportunity_score),
        "opportunity_horizon": opportunity_horizon_label(
            float(momentum_details["momentum_explosion_score"]),
            score.long_score,
        ),
        "empirical_1w_return": empirical_1w_return,
        "empirical_1w_annualized_return": annualized_return_pct(
            empirical_1w_return,
            "1 week",
        ),
        "empirical_2w_return": empirical_2w_return,
        "empirical_2w_annualized_return": annualized_return_pct(
            empirical_2w_return,
            "2 weeks",
        ),
        "empirical_4w_return": empirical_4w_return,
        "empirical_4w_annualized_return": annualized_return_pct(
            empirical_4w_return,
            "4 weeks",
        ),
        "empirical_6w_return": empirical_6w_return,
        "empirical_6w_annualized_return": annualized_return_pct(
            empirical_6w_return,
            "6 weeks",
        ),
        "empirical_1w_win_rate": empirical_1w_win_rate,
        "empirical_2w_win_rate": empirical_2w_win_rate,
        "empirical_4w_win_rate": empirical_4w_win_rate,
        "empirical_6w_win_rate": empirical_6w_win_rate,
        "empirical_1m_return": empirical_1m_return,
        "empirical_1m_annualized_return": annualized_return_pct(
            empirical_1m_return,
            "1 month",
        ),
        "empirical_1m_win_rate": empirical_1m_win_rate,
        "empirical_3m_return": empirical_3m_return,
        "empirical_3m_annualized_return": annualized_return_pct(
            empirical_3m_return,
            "3 months",
        ),
        "empirical_3m_win_rate": empirical_3m_win_rate,
        "empirical_6m_return": empirical_6m_return,
        "empirical_6m_annualized_return": annualized_return_pct(
            empirical_6m_return,
            "6 months",
        ),
        "empirical_6m_win_rate": empirical_6m_win_rate,
        "empirical_12m_return": empirical_12m_return,
        "empirical_12m_annualized_return": annualized_return_pct(
            empirical_12m_return,
            "12 months",
        ),
        "empirical_3y_return": empirical_3y_return,
        "empirical_3y_annualized_return": annualized_return_pct(
            empirical_3y_return,
            "3 years",
        ),
        "empirical_3y_win_rate": empirical_3y_win_rate,
        "empirical_5y_return": empirical_5y_return,
        "empirical_5y_annualized_return": annualized_return_pct(
            empirical_5y_return,
            "5 years",
        ),
        "empirical_5y_win_rate": empirical_5y_win_rate,
        "empirical_win_rate": empirical_win_rate,
        "confidence_pct": confidence_pct,
        "historical_downside": historical_downside,
        "risk_reward_ratio": risk_reward_ratio,
        "best_holding_period": best_holding_period_from_returns(
            empirical_12m_return,
            empirical_3y_return,
            empirical_5y_return,
        ),
        "expected_value_12m": expected_value_12m,
        "expected_value_3y": expected_value_3y,
        "expected_value_5y": expected_value_5y,
        "best_expected_value": max(
            expected_value_12m,
            expected_value_3y,
            expected_value_5y,
        ),
        "empirical_outlook": empirical_outlook_label(
            empirical_12m_return,
            empirical_5y_return,
            empirical_win_rate,
        ),
    }
    result.update(trading_advisory_fields(result))
    result.update(swing_advisory_fields(result))
    result.update(compounder_advisory_fields(result))
    for prefix in ("trading", "swing", "compounder"):
        horizon = str(result.get(f"{prefix}_horizon", recommended_horizon))
        result[f"{prefix}_forecast_confidence_band"] = forecast_confidence_band(
            empirical_forecast,
            horizon,
        )
    return result


def sort_screener_frame(frame: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_column = sort_by if sort_by in frame.columns else "overall_score"
    return frame.sort_values(sort_column, ascending=False).reset_index(drop=True)


def score_multiple_tickers(
    tickers: list[str],
    validation_records: pd.DataFrame,
    min_observations: int = 1,
    sort_by: str = "overall_score",
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        prices, is_fallback = safe_price_history(ticker, period="1y")
        score, _quality = score_for_app(ticker, prices, is_fallback)
        empirical = empirical_score_bucket_forecast(
            score.overall_score,
            validation_records,
            min_observations=min_observations,
        )
        rows.append(
            screener_row_from_score(
                score,
                empirical,
                prices,
                min_observations=min_observations,
            )
        )
    if not rows:
        return pd.DataFrame()
    frame = normalize_advisory_scores(pd.DataFrame(rows))
    frame = apply_forecast_uniqueness_scores(frame)
    return sort_screener_frame(frame, sort_by)


def best_opportunity_ticker(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    working = frame.copy()
    max_5y = max(float(working["empirical_5y_return"].max()), 1.0)
    working["opportunity_score"] = (
        working["overall_score"].astype(float) * 0.40
        + working["confidence_pct"].astype(float) * 0.30
        + (working["empirical_5y_return"].astype(float) / max_5y * 100) * 0.30
    )
    return str(working.sort_values("opportunity_score", ascending=False).iloc[0]["ticker"])


def opportunity_finder_results(
    screener_frame: pd.DataFrame,
    minimum_score: float,
    minimum_confidence: float,
    limit: int = 20,
) -> pd.DataFrame:
    if screener_frame.empty:
        return screener_frame
    filtered = screener_frame[
        (screener_frame["short_term_opportunity_score"] >= minimum_score)
        & (screener_frame["confidence_pct"] >= minimum_confidence)
    ]
    return sort_screener_frame(filtered, "master_rank_score").head(limit)


def portfolio_weights(screener_frame: pd.DataFrame, mode: str) -> pd.Series:
    if screener_frame.empty:
        return pd.Series(dtype=float)
    if mode == "Short-Term Opportunity Portfolio":
        raw_weights = screener_frame["short_term_opportunity_score"].astype(float)
    elif mode == "Balanced Portfolio":
        raw_weights = (
            screener_frame["overall_score"].astype(float) * 0.50
            + screener_frame["short_term_opportunity_score"].astype(float) * 0.50
        )
    else:
        raw_weights = screener_frame["overall_score"].astype(float)
    total = float(raw_weights.sum())
    if total <= 0:
        return pd.Series([0.0] * len(screener_frame), index=screener_frame.index)
    return raw_weights / total


def portfolio_builder_frame(
    screener_frame: pd.DataFrame,
    mode: str = "Long-Term Portfolio",
) -> pd.DataFrame:
    if screener_frame.empty:
        return pd.DataFrame()
    portfolio = screener_frame[
        [
            "ticker",
            "overall_score",
            "short_term_opportunity_score",
            "opportunity_horizon",
            "empirical_outlook",
            "expected_value_5y",
            "empirical_12m_return",
            "empirical_3y_return",
            "empirical_5y_return",
        ]
    ].copy()
    portfolio["weight"] = portfolio_weights(screener_frame, mode)
    return portfolio.rename(columns={"overall_score": "score"})[
        [
            "ticker",
            "score",
            "short_term_opportunity_score",
            "weight",
            "opportunity_horizon",
            "empirical_outlook",
            "expected_value_5y",
        ]
    ]


def portfolio_builder_summary(
    screener_frame: pd.DataFrame,
    mode: str = "Long-Term Portfolio",
) -> dict[str, object]:
    if screener_frame.empty:
        return {
            "weighted_average_score": 0.0,
            "expected_portfolio_12m_return": 0.0,
            "expected_portfolio_3y_return": 0.0,
            "expected_portfolio_5y_return": 0.0,
            "portfolio_confidence": 0.0,
            "portfolio_horizon_classification": "",
            "strongest_holding": "",
            "weakest_holding": "",
        }
    weights = portfolio_weights(screener_frame, mode)
    top_weight_index = weights.sort_values(ascending=False).index[0]
    return {
        "weighted_average_score": float((screener_frame["overall_score"] * weights).sum()),
        "expected_portfolio_12m_return": float(
            (screener_frame["empirical_12m_return"] * weights).sum()
        ),
        "expected_portfolio_3y_return": float(
            (screener_frame["empirical_3y_return"] * weights).sum()
        ),
        "expected_portfolio_5y_return": float(
            (screener_frame["empirical_5y_return"] * weights).sum()
        ),
        "portfolio_confidence": float((screener_frame["confidence_pct"] * weights).sum()),
        "portfolio_horizon_classification": str(
            screener_frame.loc[top_weight_index, "opportunity_horizon"]
        ),
        "strongest_holding": str(
            screener_frame.sort_values("overall_score", ascending=False).iloc[0]["ticker"]
        ),
        "weakest_holding": str(
            screener_frame.sort_values("overall_score", ascending=True).iloc[0]["ticker"]
        ),
    }


def best_ideas_category_specs() -> dict[str, dict[str, str]]:
    return {
        "Trading": {
            "score": "trading_score",
            "conviction": "trading_conviction_level",
            "expected_return": "trading_expected_return",
            "expected_range": "trading_expected_return_range",
            "win_rate": "trading_win_rate",
            "confidence": "confidence_pct",
            "holding_period": "trading_horizon",
        },
        "Swing": {
            "score": "swing_score",
            "conviction": "swing_conviction_level",
            "expected_return": "swing_expected_return",
            "expected_range": "swing_expected_return_range",
            "win_rate": "swing_win_rate",
            "confidence": "confidence_pct",
            "holding_period": "swing_horizon",
        },
        "Compounder": {
            "score": "compounder_score",
            "conviction": "compounder_conviction_level",
            "expected_return": "compounder_expected_return",
            "expected_range": "compounder_expected_return_range",
            "win_rate": "compounder_win_rate",
            "confidence": "confidence_pct",
            "holding_period": "compounder_horizon",
        },
    }


def best_ideas_for_today(screener_frame: pd.DataFrame) -> pd.DataFrame:
    if screener_frame.empty:
        return pd.DataFrame()
    rows = []
    for label, category, horizon_label in [
        ("Top Trade", "Trading", "1-4 weeks"),
        ("Top Swing", "Swing", "3-6 months"),
        ("Top Compounder", "Compounder", "1-5 years"),
    ]:
        spec = best_ideas_category_specs()[category]
        candidate = sort_screener_frame(screener_frame, spec["score"]).iloc[0]
        rows.append(
            {
                "Idea": label,
                "Ticker": candidate["ticker"],
                "Expected Return": candidate[spec["expected_range"]],
                "Holding Period": candidate[spec["holding_period"]],
                "Horizon": horizon_label,
                "Win Rate": candidate[spec["win_rate"]],
                "Conviction": candidate[spec["conviction"]],
            }
        )
    return pd.DataFrame(rows)


def best_ideas_portfolio_frame(
    screener_frame: pd.DataFrame,
    model: str,
    per_sleeve: int = 5,
) -> pd.DataFrame:
    if screener_frame.empty:
        return pd.DataFrame()
    allocations = BEST_IDEAS_PORTFOLIO_ALLOCATIONS.get(
        model,
        BEST_IDEAS_PORTFOLIO_ALLOCATIONS["Balanced Growth"],
    )
    specs = best_ideas_category_specs()
    rows = []
    selected_tickers: set[str] = set()
    for category, sleeve_weight in allocations.items():
        spec = specs[category]
        ranked = sort_screener_frame(screener_frame, spec["score"])
        ranked = ranked[~ranked["ticker"].isin(selected_tickers)].head(per_sleeve)
        if ranked.empty:
            ranked = sort_screener_frame(screener_frame, spec["score"]).head(per_sleeve)
        base_sizes = ranked[spec["conviction"]].map(conviction_position_size_pct)
        base_total = float(base_sizes.sum())
        if base_total <= 0:
            base_sizes = pd.Series([1.0] * len(ranked), index=ranked.index)
            base_total = float(base_sizes.sum())
        for index, item in ranked.iterrows():
            ticker = str(item["ticker"])
            selected_tickers.add(ticker)
            expected_return = float(item[spec["expected_return"]])
            downside = float(item.get("historical_downside", 0.0))
            if downside <= 0:
                downside = max(1.0, abs(expected_return) * 0.40)
            weight = float(sleeve_weight * (base_sizes.loc[index] / base_total) * 100)
            rows.append(
                {
                    "portfolio_model": model,
                    "sleeve": category,
                    "ticker": ticker,
                    "weight": weight,
                    "conviction": item[spec["conviction"]],
                    "position_size_basis": conviction_position_size_pct(
                        str(item[spec["conviction"]])
                    ),
                    "expected_return": expected_return,
                    "expected_return_range": item[spec["expected_range"]],
                    "win_rate": float(item[spec["win_rate"]]),
                    "confidence": float(item[spec["confidence"]]),
                    "holding_period": item[spec["holding_period"]],
                    "historical_downside": downside,
                }
            )
    portfolio = pd.DataFrame(rows)
    if portfolio.empty:
        return portfolio
    total_weight = float(portfolio["weight"].sum())
    if total_weight > 0:
        portfolio["weight"] = portfolio["weight"] / total_weight * 100
    portfolio["weight"] = portfolio["weight"].round(2)
    return portfolio


def best_ideas_portfolio_summary(portfolio: pd.DataFrame) -> dict[str, float]:
    if portfolio.empty:
        return {
            "expected_portfolio_return": 0.0,
            "expected_portfolio_win_rate": 0.0,
            "expected_drawdown": 0.0,
            "risk_reward_ratio": 0.0,
            "portfolio_confidence": 0.0,
        }
    weights = portfolio["weight"].astype(float) / 100
    expected_return = float((portfolio["expected_return"].astype(float) * weights).sum())
    win_rate = float((portfolio["win_rate"].astype(float) * weights).sum())
    drawdown = float((portfolio["historical_downside"].astype(float) * weights).sum())
    confidence = float((portfolio["confidence"].astype(float) * weights).sum())
    return {
        "expected_portfolio_return": expected_return,
        "expected_portfolio_win_rate": win_rate,
        "expected_drawdown": drawdown,
        "risk_reward_ratio": expected_return / drawdown if drawdown else 0.0,
        "portfolio_confidence": confidence,
    }


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
    tab_quality_edge,
    tab_universe,
    tab_opportunity_finder,
    tab_advisory,
    tab_backtest,
    tab_watchlist,
    tab_portfolio_builder,
    tab_best_ideas_portfolio,
    tab_portfolio,
) = st.tabs(
    [
        "Screener",
        "Forecast",
        "Prediction Testing",
        "Validation Dashboard",
        "Prediction Accuracy Dashboard",
        "ALPHA VALIDATION",
        "Quality of Edge",
        "Universe Backtest",
        "Opportunity Finder",
        "Advisory",
        "Backtesting",
        "Watchlist",
        "Portfolio Builder",
        "Best Ideas Portfolio",
        "Portfolio Testing",
    ]
)

with tab_screener:
    ticker_input = st.text_input("Ticker", value="AAPL").upper().strip()
    screener_sort_by = st.selectbox("Sort By", SCREENER_SORT_OPTIONS)
    if st.button("Score ticker", type="primary"):
        screener_tickers = custom_tickers_from_text(ticker_input)
        if not screener_tickers:
            st.warning("Enter at least one ticker.")
        elif len(screener_tickers) == 1:
            ticker = screener_tickers[0]
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
        else:
            validation_records = empirical_validation_records(
                tuple(screener_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            screener_frame = score_multiple_tickers(
                screener_tickers,
                validation_records,
                min_observations=1,
                sort_by=screener_sort_by,
            )
            if screener_frame.empty:
                st.warning("No screener results were generated.")
            else:
                summary_cols = st.columns(6)
                summary_cols[0].metric(
                    "Highest Score",
                    f"{screener_frame['overall_score'].max():.2f}",
                )
                summary_cols[1].metric(
                    "Average Score",
                    f"{screener_frame['overall_score'].mean():.2f}",
                )
                summary_cols[2].metric(
                    "Strong Buy Count",
                    int((screener_frame["label"] == "Strong Buy").sum()),
                )
                summary_cols[3].metric(
                    "Buy Count",
                    int((screener_frame["label"] == "Buy").sum()),
                )
                summary_cols[4].metric(
                    "Watchlist Count",
                    int((screener_frame["label"] == "Watchlist").sum()),
                )
                summary_cols[5].metric(
                    "Avoid Count",
                    int((screener_frame["label"] == "Avoid").sum()),
                )
                empirical_cols = st.columns(4)
                empirical_cols[0].metric(
                    "Average Empirical 12M Return",
                    f"{screener_frame['empirical_12m_return'].mean():.1f}%",
                )
                empirical_cols[1].metric(
                    "Average Empirical 5Y Return",
                    f"{screener_frame['empirical_5y_return'].mean():.1f}%",
                )
                empirical_cols[2].metric(
                    "Average Confidence",
                    f"{screener_frame['confidence_pct'].mean():.1f}%",
                )
                empirical_cols[3].metric(
                    "Best Opportunity",
                    best_opportunity_ticker(screener_frame),
                )
                st.dataframe(
                    screener_frame,
                    use_container_width=True,
                    column_config={
                        "expected_value_12m": st.column_config.NumberColumn(
                            "expected_value_12m",
                            format="$%d",
                        ),
                        "expected_value_3y": st.column_config.NumberColumn(
                            "expected_value_3y",
                            format="$%d",
                        ),
                        "expected_value_5y": st.column_config.NumberColumn(
                            "expected_value_5y",
                            format="$%d",
                        ),
                        "best_expected_value": st.column_config.NumberColumn(
                            "best_expected_value",
                            format="$%d",
                        ),
                    },
                )
                st.download_button(
                    "Download CSV",
                    data=screener_frame.to_csv(index=False),
                    file_name="trends_alpha_screener.csv",
                    mime="text/csv",
                )


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


with tab_quality_edge:
    quality_universe_text = st.text_area(
        "Quality universe tickers",
        value="AAPL, MSFT, META, NVDA, GOOGL, AMZN, PLTR, TSLA",
    )
    quality_start = st.text_input("Quality validation start date", value="2018-01-01")
    quality_max_tickers = st.number_input(
        "Quality maximum tickers",
        min_value=1,
        max_value=100,
        value=8,
        step=1,
    )
    if st.button("Run quality of edge analysis"):
        quality_tickers = custom_tickers_from_text(quality_universe_text)[: int(quality_max_tickers)]
        if not quality_tickers:
            st.warning("Add at least one ticker.")
        else:
            prices, fallback_flags = load_price_histories_for_tickers(
                quality_tickers,
                start="2013-01-01",
            )
            validation = alpha_validation_frame(
                prices,
                start_date=quality_start,
                fallback_data_used_by_ticker=fallback_flags,
                step_days=252,
            )
            benchmarks = {
                "S&P 500": benchmark_return_frame(
                    "S&P 500",
                    sample_price_history(start="2013-01-01", end="2026-01-01"),
                    start_date=quality_start,
                    step_days=252,
                ),
                "Nasdaq 100": benchmark_return_frame(
                    "Nasdaq 100",
                    sample_price_history(start="2013-01-01", end="2026-01-01"),
                    start_date=quality_start,
                    step_days=252,
                ),
            }
            top_decile = top_decile_test(validation, benchmarks)
            top_20 = top_20_portfolio_test(validation, benchmarks)
            regime = regime_analysis(validation)
            false_positive = false_positive_analysis(validation)
            metrics = quality_of_edge_metrics(validation, benchmarks)
            quality_empirical_records = empirical_validation_records(
                tuple(quality_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            quality_forecast_frame = score_multiple_tickers(
                quality_tickers,
                quality_empirical_records,
                min_observations=1,
                sort_by="forecast_uniqueness_score",
            )
            quality_forecast_differentiation = forecast_uniqueness_ratio(
                quality_forecast_frame
            )

            st.subheader("Quality Verdict")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Final Verdict", metrics["final_verdict"])
            metric_cols[1].metric(
                "Alpha Consistency Score",
                f"{metrics['alpha_consistency_score']:.1f}",
            )
            metric_cols[2].metric(
                "Consistency Label",
                metrics["alpha_consistency_label"],
            )
            metric_cols[3].metric(
                "False Positive Rate",
                f"{metrics['false_positive_rate_pct']:.1f}%",
            )
            evidence_cols = st.columns(4)
            evidence_cols[0].metric(
                "Top Decile Alpha",
                f"{metrics['top_decile_alpha_pct']:.1f}%",
            )
            evidence_cols[1].metric(
                "Top 20 Alpha",
                f"{metrics['top_20_alpha_pct']:.1f}%",
            )
            evidence_cols[2].metric(
                "Forecast/Actual Correlation",
                f"{metrics['forecast_actual_correlation']:.2f}",
            )
            evidence_cols[3].metric(
                "Calibration Accuracy",
                f"{metrics['calibration_accuracy_pct']:.1f}%",
            )
            forecast_cols = st.columns(2)
            forecast_cols[0].metric(
                "Forecast Differentiation",
                f"{quality_forecast_differentiation:.1f}%",
            )
            forecast_cols[1].metric(
                "Forecast Bucketing Flag",
                yes_no(
                    not quality_forecast_frame.empty
                    and bool(quality_forecast_frame["forecast_bucketing_flag"].any())
                ),
            )
            if (
                not quality_forecast_frame.empty
                and bool(quality_forecast_frame["forecast_bucketing_flag"].any())
            ):
                st.warning(
                    "Forecast bucketing detected in: "
                    f"{quality_forecast_frame['forecast_bucketed_columns'].iloc[0]}"
                )

            st.subheader("Top Decile Performance")
            st.dataframe(top_decile, use_container_width=True)
            st.subheader("Top 20 Portfolio Performance")
            st.dataframe(top_20, use_container_width=True)
            st.subheader("Regime Analysis")
            st.dataframe(regime, use_container_width=True)
            st.subheader("False Positive Analysis")
            st.dataframe(pd.DataFrame([false_positive]), use_container_width=True)
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


with tab_opportunity_finder:
    opportunity_universe = st.selectbox(
        "Opportunity universe",
        ["S&P 500", "Nasdaq 100", "Custom tickers"],
    )
    opportunity_custom = st.text_area(
        "Custom opportunity tickers",
        value="AAPL,MSFT,META,NVDA,GOOGL,AMZN,PLTR,TSLA",
    )
    opportunity_max = st.number_input(
        "Maximum stocks",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )
    opportunity_min_score = st.number_input(
        "Minimum score",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=5.0,
    )
    opportunity_min_confidence = st.number_input(
        "Minimum confidence",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=5.0,
    )
    if st.button("Find opportunities"):
        if opportunity_universe == "S&P 500":
            opportunity_tickers = get_universe("sp500")
        elif opportunity_universe == "Nasdaq 100":
            opportunity_tickers = get_universe("nasdaq100")
        else:
            opportunity_tickers = custom_tickers_from_text(opportunity_custom)
        opportunity_tickers = opportunity_tickers[: int(opportunity_max)]
        if not opportunity_tickers:
            st.warning("Add at least one ticker to scan.")
        else:
            validation_records = empirical_validation_records(
                tuple(opportunity_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            opportunity_frame = score_multiple_tickers(
                opportunity_tickers,
                validation_records,
                min_observations=1,
                sort_by="master_rank_score",
            )
            opportunity_frame = opportunity_finder_results(
                opportunity_frame,
                minimum_score=opportunity_min_score,
                minimum_confidence=opportunity_min_confidence,
                limit=20,
            )
            if opportunity_frame.empty:
                st.warning("No opportunities matched the current filters.")
            else:
                top_cols = st.columns(3)
                top_cols[0].metric(
                    "Best Short-Term Opportunity",
                    opportunity_frame.iloc[0]["ticker"],
                )
                top_cols[1].metric(
                    "Best Long-Term Opportunity",
                    opportunity_frame.sort_values("overall_score", ascending=False).iloc[0][
                        "ticker"
                    ],
                )
                top_cols[2].metric(
                    "Highest Momentum Stock",
                    opportunity_frame.sort_values(
                        "momentum_explosion_score",
                        ascending=False,
                    ).iloc[0]["ticker"],
                )
                more_cols = st.columns(3)
                more_cols[0].metric(
                    "Highest Confidence Stock",
                    opportunity_frame.sort_values("confidence_pct", ascending=False).iloc[0][
                        "ticker"
                    ],
                )
                more_cols[1].metric(
                    "Highest Expected 12M Return",
                    opportunity_frame.sort_values(
                        "empirical_12m_return",
                        ascending=False,
                    ).iloc[0]["ticker"],
                )
                more_cols[2].metric(
                    "Highest Expected 5Y Return",
                    opportunity_frame.sort_values(
                        "empirical_5y_return",
                        ascending=False,
                    ).iloc[0]["ticker"],
                )

                st.subheader("Top 20 Opportunities")
                finder_display = opportunity_frame[
                    [
                        "ticker",
                        "master_rank_score",
                        "alpha_consistency_score",
                        "overall_score",
                        "short_term_opportunity_score",
                        "momentum_explosion_score",
                        "opportunity_horizon",
                        "empirical_1m_return",
                        "empirical_12m_return",
                        "confidence_pct",
                    ]
                ]
                st.dataframe(finder_display, use_container_width=True)
                st.download_button(
                    "Download Opportunities CSV",
                    data=opportunity_frame.to_csv(index=False),
                    file_name="trends_alpha_opportunities.csv",
                    mime="text/csv",
                )
        st.caption(ADVICE_WARNING)


with tab_advisory:
    advisory_tickers_text = st.text_area(
        "Advisory tickers",
        value="AAPL,MSFT,META,NVDA,GOOGL,AMZN,PLTR,TSLA",
    )
    advisory_max = st.number_input(
        "Advisory maximum tickers",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )
    advisory_min_observations = st.number_input(
        "Minimum advisory observations",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
    )
    if st.button("Run research advisory"):
        advisory_tickers = custom_tickers_from_text(advisory_tickers_text)[: int(advisory_max)]
        if not advisory_tickers:
            st.warning("Add at least one ticker.")
        else:
            validation_records = empirical_validation_records(
                tuple(advisory_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            advisory_frame = score_multiple_tickers(
                advisory_tickers,
                validation_records,
                min_observations=int(advisory_min_observations),
                sort_by="advisory_score",
            )
            if advisory_frame.empty:
                st.warning("No advisory results were generated.")
            else:
                forecast_differentiation = forecast_uniqueness_ratio(advisory_frame)
                trading_frame = sort_screener_frame(
                    advisory_frame,
                    "trading_score",
                ).head(10)
                swing_frame = sort_screener_frame(advisory_frame, "swing_score").head(10)
                compounder_frame = sort_screener_frame(
                    advisory_frame,
                    "compounder_score",
                ).head(10)

                st.subheader("Research Advisory")
                advisory_metrics = st.columns(2)
                advisory_metrics[0].metric(
                    "Forecast Differentiation",
                    f"{forecast_differentiation:.1f}%",
                )
                advisory_metrics[1].metric(
                    "Forecast Bucketing Flag",
                    yes_no(bool(advisory_frame["forecast_bucketing_flag"].any())),
                )
                if bool(advisory_frame["forecast_bucketing_flag"].any()):
                    st.warning(
                        "Forecast bucketing detected in: "
                        f"{advisory_frame['forecast_bucketed_columns'].iloc[0]}"
                    )
                display = advisory_frame[
                    [
                        "ticker",
                        "advisory_action",
                        "recommended_holding_period",
                        "expected_return_range",
                        "expected_benchmark_return",
                        "expected_alpha",
                        "historical_win_rate",
                        "confidence_level",
                        "forecast_confidence_band",
                        "recommended_horizon_score",
                        "risk_adjusted_return",
                        "duration_penalty",
                        "advisory_percentile",
                        "conviction_level",
                        "position_size_guidance",
                        "forecast_uniqueness_score",
                        "forecast_differentiation_pct",
                        "forecast_bucketing_flag",
                        "advisory_score",
                    ]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "advisory_action": "Recommended Action",
                        "recommended_holding_period": "Holding Period",
                        "expected_return_range": "Expected Return",
                        "expected_benchmark_return": "Benchmark Return",
                        "expected_alpha": "Expected Alpha",
                        "historical_win_rate": "Historical Win Rate",
                        "confidence_level": "Confidence",
                        "forecast_confidence_band": "Forecast Band",
                        "recommended_horizon_score": "Horizon Score",
                        "risk_adjusted_return": "Risk-Adjusted Return",
                        "duration_penalty": "Duration Penalty",
                        "advisory_percentile": "Percentile",
                        "conviction_level": "Conviction",
                        "position_size_guidance": "Position Size",
                        "forecast_uniqueness_score": "Forecast Uniqueness",
                        "forecast_differentiation_pct": "Forecast Differentiation %",
                        "forecast_bucketing_flag": "Bucket Flag",
                        "advisory_score": "Advisory Score",
                    }
                )
                st.dataframe(display, use_container_width=True)

                st.subheader("Best Ideas")
                best_ideas = pd.DataFrame(
                    [
                        {
                            "Idea": "Best Trade (1-4 weeks)",
                            "Ticker": trading_frame.iloc[0]["ticker"],
                            "Expected Return": trading_frame.iloc[0][
                                "trading_expected_return_range"
                            ],
                            "Win Rate": trading_frame.iloc[0]["trading_win_rate"],
                            "Confidence": trading_frame.iloc[0]["confidence_level"],
                            "Position Size": trading_frame.iloc[0][
                                "trading_position_size_guidance"
                            ],
                            "Conviction": trading_frame.iloc[0]["trading_conviction_level"],
                        },
                        {
                            "Idea": "Best Swing (3-6 months)",
                            "Ticker": swing_frame.iloc[0]["ticker"],
                            "Expected Return": swing_frame.iloc[0][
                                "swing_expected_return_range"
                            ],
                            "Win Rate": swing_frame.iloc[0]["swing_win_rate"],
                            "Confidence": swing_frame.iloc[0]["confidence_level"],
                            "Position Size": swing_frame.iloc[0][
                                "swing_position_size_guidance"
                            ],
                            "Conviction": swing_frame.iloc[0]["swing_conviction_level"],
                        },
                        {
                            "Idea": "Best Compounder (1-5 years)",
                            "Ticker": compounder_frame.iloc[0]["ticker"],
                            "Expected Return": compounder_frame.iloc[0][
                                "compounder_expected_return_range"
                            ],
                            "Win Rate": compounder_frame.iloc[0]["compounder_win_rate"],
                            "Confidence": compounder_frame.iloc[0]["confidence_level"],
                            "Position Size": compounder_frame.iloc[0][
                                "compounder_position_size_guidance"
                            ],
                            "Conviction": compounder_frame.iloc[0][
                                "compounder_conviction_level"
                            ],
                        },
                    ]
                )
                st.dataframe(best_ideas, use_container_width=True)

                st.subheader("Top Trading Opportunities")
                trading_display = trading_frame[
                    [
                        "ticker",
                        "trading_action",
                        "trading_expected_return_range",
                        "trading_expected_benchmark_return",
                        "trading_expected_alpha",
                        "trading_expected_annualized_return",
                        "trading_win_rate",
                        "confidence_level",
                        "trading_forecast_confidence_band",
                        "trading_conviction_level",
                        "trading_position_size_guidance",
                        "trading_horizon",
                        "trading_percentile",
                        "trading_score",
                    ]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "trading_action": "Recommendation",
                        "trading_expected_return_range": "Expected Return Range",
                        "trading_expected_benchmark_return": "Benchmark Return",
                        "trading_expected_alpha": "Expected Alpha",
                        "trading_expected_annualized_return": "Annualized Return",
                        "trading_win_rate": "Historical Win Rate",
                        "confidence_level": "Confidence",
                        "trading_forecast_confidence_band": "Forecast Band",
                        "trading_conviction_level": "Conviction",
                        "trading_position_size_guidance": "Position Size",
                        "trading_horizon": "Suggested Holding Period",
                        "trading_percentile": "Percentile",
                        "trading_score": "Trading Score",
                    }
                )
                st.dataframe(trading_display, use_container_width=True)
                st.download_button(
                    "Download Trading Advisory CSV",
                    data=trading_frame.to_csv(index=False),
                    file_name="trends_alpha_trading_advisory.csv",
                    mime="text/csv",
                )

                st.subheader("Top Swing Opportunities")
                swing_display = swing_frame[
                    [
                        "ticker",
                        "swing_action",
                        "swing_expected_return_range",
                        "swing_expected_benchmark_return",
                        "swing_expected_alpha",
                        "swing_expected_annualized_return",
                        "swing_win_rate",
                        "confidence_level",
                        "swing_forecast_confidence_band",
                        "swing_conviction_level",
                        "swing_position_size_guidance",
                        "swing_horizon",
                        "swing_percentile",
                        "swing_score",
                    ]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "swing_action": "Recommendation",
                        "swing_expected_return_range": "Expected Return Range",
                        "swing_expected_benchmark_return": "Benchmark Return",
                        "swing_expected_alpha": "Expected Alpha",
                        "swing_expected_annualized_return": "Annualized Return",
                        "swing_win_rate": "Historical Win Rate",
                        "confidence_level": "Confidence",
                        "swing_forecast_confidence_band": "Forecast Band",
                        "swing_conviction_level": "Conviction",
                        "swing_position_size_guidance": "Position Size",
                        "swing_horizon": "Suggested Holding Period",
                        "swing_percentile": "Percentile",
                        "swing_score": "Swing Score",
                    }
                )
                st.dataframe(swing_display, use_container_width=True)
                st.download_button(
                    "Download Swing Advisory CSV",
                    data=swing_frame.to_csv(index=False),
                    file_name="trends_alpha_swing_advisory.csv",
                    mime="text/csv",
                )

                st.subheader("Top Long-Term Compounders")
                compounder_display = compounder_frame[
                    [
                        "ticker",
                        "compounder_action",
                        "compounder_expected_return_range",
                        "compounder_expected_benchmark_return",
                        "compounder_expected_alpha",
                        "compounder_expected_annualized_return",
                        "compounder_win_rate",
                        "confidence_level",
                        "compounder_forecast_confidence_band",
                        "compounder_conviction_level",
                        "compounder_position_size_guidance",
                        "compounder_horizon",
                        "compounder_percentile",
                        "compounder_score",
                    ]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "compounder_action": "Recommendation",
                        "compounder_expected_return_range": "Expected Return Range",
                        "compounder_expected_benchmark_return": "Benchmark Return",
                        "compounder_expected_alpha": "Expected Alpha",
                        "compounder_expected_annualized_return": "Annualized Return",
                        "compounder_win_rate": "Historical Win Rate",
                        "confidence_level": "Confidence",
                        "compounder_forecast_confidence_band": "Forecast Band",
                        "compounder_conviction_level": "Conviction",
                        "compounder_position_size_guidance": "Position Size",
                        "compounder_horizon": "Suggested Holding Period",
                        "compounder_percentile": "Percentile",
                        "compounder_score": "Compounder Score",
                    }
                )
                st.dataframe(compounder_display, use_container_width=True)
                st.download_button(
                    "Download Compounder Advisory CSV",
                    data=compounder_frame.to_csv(index=False),
                    file_name="trends_alpha_compounder_advisory.csv",
                    mime="text/csv",
                )

                st.subheader("Top 10 Blended Advisory Summaries")
                for summary in advisory_frame["advisory_summary"].head(10):
                    st.write(summary)
                st.download_button(
                    "Download Blended Advisory CSV",
                    data=advisory_frame.to_csv(index=False),
                    file_name="trends_alpha_research_advisory.csv",
                    mime="text/csv",
                )
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


with tab_portfolio_builder:
    builder_input = st.text_area(
        "Portfolio tickers",
        value="AAPL,MSFT,META,NVDA,GOOGL,AMZN",
    )
    portfolio_mode = st.selectbox(
        "Portfolio mode",
        [
            "Long-Term Portfolio",
            "Short-Term Opportunity Portfolio",
            "Balanced Portfolio",
        ],
    )
    if st.button("Build portfolio"):
        builder_tickers = custom_tickers_from_text(builder_input)
        if not builder_tickers:
            st.warning("Enter at least one ticker.")
        else:
            validation_records = empirical_validation_records(
                tuple(builder_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            screener_frame = score_multiple_tickers(
                builder_tickers,
                validation_records,
                min_observations=1,
                sort_by="overall_score",
            )
            portfolio_frame = portfolio_builder_frame(screener_frame, mode=portfolio_mode)
            summary = portfolio_builder_summary(screener_frame, mode=portfolio_mode)

            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Weighted Average Score",
                f"{summary['weighted_average_score']:.2f}",
            )
            summary_cols[1].metric(
                "Expected Portfolio 12M Return",
                f"{summary['expected_portfolio_12m_return']:.1f}%",
            )
            summary_cols[2].metric(
                "Expected Portfolio 3Y Return",
                f"{summary['expected_portfolio_3y_return']:.1f}%",
            )
            holding_cols = st.columns(3)
            holding_cols[0].metric(
                "Expected Portfolio 5Y Return",
                f"{summary['expected_portfolio_5y_return']:.1f}%",
            )
            holding_cols[1].metric(
                "Portfolio Confidence",
                f"{summary['portfolio_confidence']:.1f}%",
            )
            holding_cols[2].metric(
                "Portfolio Horizon",
                summary["portfolio_horizon_classification"],
            )
            strength_cols = st.columns(2)
            strength_cols[0].metric("Strongest Holding", summary["strongest_holding"])
            strength_cols[1].metric("Weakest Holding", summary["weakest_holding"])

            st.dataframe(
                portfolio_frame,
                use_container_width=True,
                column_config={
                    "weight": st.column_config.NumberColumn("weight", format="%.4f"),
                    "expected_value_5y": st.column_config.NumberColumn(
                        "expected_value_5y",
                        format="$%d",
                    ),
                },
            )
            st.download_button(
                "Download Portfolio CSV",
                data=portfolio_frame.to_csv(index=False),
                file_name="trends_alpha_portfolio_builder.csv",
                mime="text/csv",
            )
        st.caption(ADVICE_WARNING)


with tab_best_ideas_portfolio:
    best_portfolio_input = st.text_area(
        "Best ideas tickers",
        value="AAPL,MSFT,META,NVDA,GOOGL,AMZN,PLTR,TSLA",
    )
    best_portfolio_model = st.selectbox(
        "Model portfolio",
        [
            "Aggressive Growth",
            "Balanced Growth",
            "Conservative Compounder",
        ],
    )
    best_portfolio_max = st.number_input(
        "Maximum tickers to scan",
        min_value=3,
        max_value=100,
        value=20,
        step=1,
    )
    if st.button("Build best ideas portfolio"):
        best_portfolio_tickers = custom_tickers_from_text(best_portfolio_input)[
            : int(best_portfolio_max)
        ]
        if not best_portfolio_tickers:
            st.warning("Enter at least one ticker.")
        else:
            validation_records = empirical_validation_records(
                tuple(best_portfolio_tickers),
                start_date="2016-01-01",
                price_start="2013-01-01",
                step_days=252,
            )
            screener_frame = score_multiple_tickers(
                best_portfolio_tickers,
                validation_records,
                min_observations=1,
                sort_by="master_rank_score",
            )
            portfolio_frame = best_ideas_portfolio_frame(
                screener_frame,
                best_portfolio_model,
            )
            summary = best_ideas_portfolio_summary(portfolio_frame)
            forecast_differentiation = forecast_uniqueness_ratio(screener_frame)
            if portfolio_frame.empty:
                st.warning("No portfolio recommendations were generated.")
            else:
                st.subheader("What to Buy Today")
                st.dataframe(best_ideas_for_today(screener_frame), use_container_width=True)

                st.subheader(best_portfolio_model)
                metric_cols = st.columns(6)
                metric_cols[0].metric(
                    "Expected Portfolio Return",
                    f"{summary['expected_portfolio_return']:.1f}%",
                )
                metric_cols[1].metric(
                    "Expected Portfolio Win Rate",
                    f"{summary['expected_portfolio_win_rate']:.1f}%",
                )
                metric_cols[2].metric(
                    "Expected Drawdown",
                    f"{summary['expected_drawdown']:.1f}%",
                )
                metric_cols[3].metric(
                    "Risk/Reward Ratio",
                    f"{summary['risk_reward_ratio']:.2f}",
                )
                metric_cols[4].metric(
                    "Portfolio Confidence",
                    f"{summary['portfolio_confidence']:.1f}%",
                )
                metric_cols[5].metric(
                    "Forecast Differentiation",
                    f"{forecast_differentiation:.1f}%",
                )
                if bool(screener_frame["forecast_bucketing_flag"].any()):
                    st.warning(
                        "Forecast bucketing detected in: "
                        f"{screener_frame['forecast_bucketed_columns'].iloc[0]}"
                    )

                display = portfolio_frame[
                    [
                        "sleeve",
                        "ticker",
                        "weight",
                        "conviction",
                        "expected_return_range",
                        "win_rate",
                        "confidence",
                        "holding_period",
                    ]
                ].rename(
                    columns={
                        "sleeve": "Sleeve",
                        "ticker": "Ticker",
                        "weight": "Weight",
                        "conviction": "Conviction",
                        "expected_return_range": "Expected Return",
                        "win_rate": "Win Rate",
                        "confidence": "Confidence",
                        "holding_period": "Holding Period",
                    }
                )
                st.dataframe(display, use_container_width=True)
                st.download_button(
                    "Download Best Ideas Portfolio CSV",
                    data=portfolio_frame.to_csv(index=False),
                    file_name="trends_alpha_best_ideas_portfolio.csv",
                    mime="text/csv",
                )
        st.caption(ADVICE_WARNING)


with tab_portfolio:
    universe_name = st.selectbox("Universe", ["sp500", "nasdaq100", "russell2000"])
    n = st.selectbox("Portfolio size", [10, 25, 50])
    st.write("Sample universe tickers:", ", ".join(get_universe(universe_name)))
    st.info(f"Top {n} hypothetical portfolio testing is scaffolded in `tae.portfolio`.")
