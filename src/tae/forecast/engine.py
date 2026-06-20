from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tae.scoring.components import clip
from tae.scoring.engine import ScoreResult

FORECAST_HORIZONS = {
    "1 month": 21,
    "3 months": 63,
    "12 months": 252,
    "3 years": 756,
    "5 years": 1260,
}

# Annualized historical factor premia priors. These are deliberately conservative
# research priors that map factor strength to future return expectations.
FACTOR_PREMIA = {
    "momentum": 0.10,
    "valuation": 0.05,
    "growth": 0.08,
    "quality": 0.07,
}


@dataclass(frozen=True)
class ForecastResult:
    ticker: str
    rows: list[dict[str, float | str]]
    confidence_score: float
    factor_exposures: dict[str, float]


def _component_score(score: ScoreResult, model: str, component_name: str) -> float | None:
    for component in score.components.get(model, []):
        if component["name"] == component_name and component["weight"]:
            return float(component["score"]) / float(component["weight"])
    return None


def factor_exposures(score: ScoreResult) -> dict[str, float]:
    valuation = _component_score(
        score,
        "medium_term_alpha",
        "Valuation Reasonableness",
    )
    growth = (score.medium_score / 100) * 0.65 + (score.long_score / 100) * 0.35
    quality = score.long_score / 100
    momentum = score.short_score / 100
    return {
        "momentum": round(clip(momentum), 4),
        "valuation": round(clip(valuation if valuation is not None else 0.45), 4),
        "growth": round(clip(growth), 4),
        "quality": round(clip(quality), 4),
    }


def annualized_base_return(score: ScoreResult) -> float:
    exposures = factor_exposures(score)
    market_prior = 0.08
    factor_alpha = sum(
        (exposures[factor] - 0.5) * premium
        for factor, premium in FACTOR_PREMIA.items()
    )
    risk_penalty = (score.risk_score / 100) * 0.12
    auxiliary_boost = (
        (score.narrative_score + score.capital_flow_score + score.surprise_score) / 30
    ) * 0.04
    return float(clip(market_prior + factor_alpha + auxiliary_boost - risk_penalty, -0.35, 0.65))


def confidence_score(score: ScoreResult, price_history: pd.DataFrame) -> float:
    quality = score.data_quality
    missing_count = len(quality.get("missing_metrics", []))
    data_points = min(len(price_history), 756)
    history_score = clip(data_points / 756) * 30
    fundamental_score = 30 if quality.get("fundamental_data_available") else 0
    live_data_score = 15 if quality.get("live_price_data_available") else 5
    sample_penalty = 15 if quality.get("sample_fundamentals_used") else 0
    fallback_penalty = 10 if quality.get("fallback_data_used") else 0
    missing_penalty = min(missing_count * 2.5, 25)
    risk_penalty = clip(score.risk_score / 100) * 20
    confidence = (
        35
        + history_score
        + fundamental_score
        + live_data_score
        - sample_penalty
        - fallback_penalty
        - missing_penalty
        - risk_penalty
    )
    return round(clip(confidence, 0, 100), 2)


def horizon_return(annualized_return: float, trading_days: int) -> float:
    years = trading_days / 252
    return (1 + annualized_return) ** years - 1


def forecast_from_score(score: ScoreResult, price_history: pd.DataFrame) -> ForecastResult:
    annual_return = annualized_base_return(score)
    confidence = confidence_score(score, price_history)
    volatility = _realized_volatility(price_history)
    confidence_width = (1 - confidence / 100) * 0.35
    rows = []
    for label, trading_days in FORECAST_HORIZONS.items():
        base = horizon_return(annual_return, trading_days)
        horizon_volatility = volatility * (trading_days / 252) ** 0.5
        spread = horizon_volatility * 0.55 + confidence_width * (trading_days / 252) ** 0.5
        rows.append(
            {
                "horizon": label,
                "bear_case_return": round((base - spread) * 100, 2),
                "base_case_return": round(base * 100, 2),
                "bull_case_return": round((base + spread) * 100, 2),
            }
        )
    return ForecastResult(
        ticker=score.ticker,
        rows=rows,
        confidence_score=confidence,
        factor_exposures=factor_exposures(score),
    )


def _realized_volatility(price_history: pd.DataFrame) -> float:
    if price_history.empty or "close" not in price_history:
        return 0.30
    returns = price_history.sort_values("date")["close"].astype(float).pct_change().dropna()
    if returns.empty:
        return 0.30
    volatility = float(returns.tail(252).std() * (252**0.5))
    return clip(volatility, 0.08, 0.90)

