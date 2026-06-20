from __future__ import annotations

from math import erf, sqrt

import pandas as pd

from tae.forecasting.models import (
    FACTOR_LABELS,
    HORIZON_FACTOR_COEFFICIENTS,
    HORIZON_GROUPS,
    ForecastDriver,
    ForecastLine,
    ForecastReport,
    ValuationEstimate,
)
from tae.scoring.components import clip
from tae.scoring.engine import ScoreResult


def build_forecast_report(score: ScoreResult, price_history: pd.DataFrame) -> ForecastReport:
    exposures = factor_exposures(score)
    volatility = realized_volatility(price_history)
    confidence = confidence_pct(score, price_history)
    forecasts = []
    annual_return = expected_return("12 months", exposures)

    for group, horizons in HORIZON_GROUPS.items():
        for horizon in horizons:
            if horizon == "3 year CAGR":
                base = annual_return * 0.92
            elif horizon == "5 year CAGR":
                base = annual_return * 0.86
            else:
                base = expected_return(horizon, exposures)
            spread = forecast_spread(horizon, volatility, confidence)
            forecasts.append(
                ForecastLine(
                    group=group,
                    horizon=horizon,
                    expected_return_pct=round(base * 100, 2),
                    bear_case_pct=round((base - spread) * 100, 2),
                    base_case_pct=round(base * 100, 2),
                    bull_case_pct=round((base + spread) * 100, 2),
                    confidence_pct=round(confidence, 2),
                )
            )

    current_price = latest_price(price_history)
    fair_value = current_price * (1 + annual_return)
    upside = (fair_value / current_price - 1) if current_price else 0.0
    probability = probability_positive(annual_return, volatility)
    valuation = ValuationEstimate(
        current_price=round(current_price, 2),
        estimated_fair_value=round(fair_value, 2),
        upside_downside_pct=round(upside * 100, 2),
        probability_positive_return_pct=round(probability * 100, 2),
        risk_rating=risk_rating(score.risk_score),
        suggested_label=suggested_label(upside, probability, score.risk_score),
    )
    positives, negatives = forecast_drivers("12 months", exposures)

    return ForecastReport(
        ticker=score.ticker,
        forecasts=forecasts,
        valuation=valuation,
        top_positive_drivers=positives,
        top_negative_drivers=negatives,
        factor_exposures={key: round(value * 100, 2) for key, value in exposures.items()},
        data_quality=score.data_quality,
    )


def factor_exposures(score: ScoreResult) -> dict[str, float]:
    component = component_exposure
    return {
        "momentum": clip(score.short_score / 100),
        "volume": component(score, "short_term_alpha", "Volume Surge", default=0.45),
        "relative_strength": average_present(
            [
                component(score, "short_term_alpha", "Relative Strength"),
                component(score, "medium_term_alpha", "Relative Strength"),
            ],
            default=0.45,
        ),
        "sector_momentum": average_present(
            [
                component(score, "short_term_alpha", "Sector Momentum"),
                component(score, "medium_term_alpha", "Sector Momentum"),
            ],
            default=0.45,
        ),
        "growth": average_present(
            [
                component(score, "medium_term_alpha", "Revenue Growth"),
                component(score, "medium_term_alpha", "EPS Growth"),
                component(score, "medium_term_alpha", "Margin Expansion"),
                component(score, "long_term_compounder", "Revenue CAGR"),
                component(score, "long_term_compounder", "EPS CAGR"),
                component(score, "long_term_compounder", "Free Cash Flow Growth"),
            ],
            default=0.35,
        ),
        "valuation": component(
            score,
            "medium_term_alpha",
            "Valuation Reasonableness",
            default=0.45,
        ),
        "quality": average_present(
            [
                component(score, "long_term_compounder", "ROIC"),
                component(score, "long_term_compounder", "Balance Sheet Quality"),
                component(score, "long_term_compounder", "Competitive Moat"),
                component(score, "long_term_compounder", "Management Quality"),
            ],
            default=0.35,
        ),
        "institutional": average_present(
            [
                component(score, "short_term_alpha", "Analyst Revisions"),
                component(score, "short_term_alpha", "Institutional Accumulation"),
                component(score, "medium_term_alpha", "Institutional Buying"),
                component(score, "medium_term_alpha", "Guidance Upgrades"),
            ],
            default=0.35,
        ),
        "risk": clip(score.risk_score / 100),
    }


def component_exposure(
    score: ScoreResult,
    model_key: str,
    component_name: str,
    default: float | None = None,
) -> float | None:
    for component in score.components.get(model_key, []):
        weight = float(component.get("weight") or 0)
        if component.get("name") == component_name and weight > 0:
            return clip(float(component.get("score") or 0) / weight)
    return default


def average_present(values: list[float | None], default: float) -> float:
    present = [value for value in values if value is not None]
    if not present:
        return default
    return clip(sum(present) / len(present))


def expected_return(horizon: str, exposures: dict[str, float]) -> float:
    coefficients = HORIZON_FACTOR_COEFFICIENTS[horizon]
    base = {
        "1 week": 0.0015,
        "1 month": 0.006,
        "3 months": 0.018,
        "6 months": 0.035,
        "12 months": 0.075,
    }[horizon]
    factor_return = sum(
        coefficients[factor] * (exposures[factor] - 0.5)
        for factor in coefficients
        if factor != "risk"
    )
    risk_return = coefficients["risk"] * exposures["risk"]
    return float(clip(base + factor_return + risk_return, -0.80, 1.50))


def confidence_pct(score: ScoreResult, price_history: pd.DataFrame) -> float:
    quality = score.data_quality
    missing_count = len(quality.get("missing_metrics", []))
    history_score = clip(len(price_history) / 756) * 28
    fundamental_score = 24 if quality.get("fundamental_data_available") else 0
    live_score = 14 if quality.get("live_price_data_available") else 6
    sample_penalty = 10 if quality.get("sample_fundamentals_used") else 0
    fallback_penalty = 8 if quality.get("fallback_data_used") else 0
    missing_penalty = min(missing_count * 2.25, 25)
    risk_penalty = clip(score.risk_score / 100) * 18
    return round(
        clip(
            38
            + history_score
            + fundamental_score
            + live_score
            - sample_penalty
            - fallback_penalty
            - missing_penalty
            - risk_penalty,
            0,
            100,
        ),
        2,
    )


def forecast_spread(horizon: str, volatility: float, confidence: float) -> float:
    year_fraction = {
        "1 week": 5 / 252,
        "1 month": 21 / 252,
        "3 months": 63 / 252,
        "6 months": 126 / 252,
        "12 months": 1.0,
        "3 year CAGR": 1.0,
        "5 year CAGR": 1.0,
    }[horizon]
    uncertainty = (1 - confidence / 100) * 0.18
    return volatility * sqrt(year_fraction) * 0.55 + uncertainty


def forecast_drivers(
    horizon: str,
    exposures: dict[str, float],
    limit: int = 5,
) -> tuple[list[ForecastDriver], list[ForecastDriver]]:
    coefficients = HORIZON_FACTOR_COEFFICIENTS["12 months" if "CAGR" in horizon else horizon]
    drivers = []
    for factor, coefficient in coefficients.items():
        contribution = coefficient * (exposures[factor] - 0.5)
        if factor == "risk":
            contribution = coefficient * exposures[factor]
        drivers.append(
            ForecastDriver(
                factor=FACTOR_LABELS[factor],
                contribution_pct=round(contribution * 100, 2),
                exposure_pct=round(exposures[factor] * 100, 2),
            )
        )
    positives = sorted(drivers, key=lambda driver: driver.contribution_pct, reverse=True)
    negatives = sorted(drivers, key=lambda driver: driver.contribution_pct)
    return positives[:limit], negatives[:limit]


def realized_volatility(price_history: pd.DataFrame) -> float:
    if price_history.empty or "close" not in price_history:
        return 0.30
    returns = price_history.sort_values("date")["close"].astype(float).pct_change().dropna()
    if returns.empty:
        return 0.30
    return clip(float(returns.tail(252).std() * sqrt(252)), 0.08, 0.95)


def latest_price(price_history: pd.DataFrame) -> float:
    if price_history.empty or "close" not in price_history:
        return 0.0
    return float(price_history.sort_values("date")["close"].iloc[-1])


def probability_positive(expected: float, volatility: float) -> float:
    if volatility <= 0:
        return 0.5
    z_score = expected / volatility
    return clip(0.5 * (1 + erf(z_score / sqrt(2))))


def risk_rating(risk_score: float) -> str:
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


def suggested_label(upside: float, probability: float, risk_score: float) -> str:
    if upside >= 0.25 and probability >= 0.62 and risk_score <= 55:
        return "Strong Buy"
    if upside >= 0.12 and probability >= 0.56 and risk_score <= 65:
        return "Buy"
    if upside >= 0.02 and probability >= 0.50 and risk_score <= 80:
        return "Watchlist"
    return "Avoid"

