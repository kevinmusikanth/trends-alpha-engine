from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tae.scoring.components import clip
from tae.scoring.fundamentals import build_feature_set, missing_metrics
from tae.scoring.models import score_model_a, score_model_b, score_model_c, validate_weights


@dataclass(frozen=True)
class ScoreResult:
    ticker: str
    short_score: float
    medium_score: float
    long_score: float
    narrative_score: float
    capital_flow_score: float
    surprise_score: float
    risk_score: float
    overall_score: float
    recommendation: str
    components: dict[str, list[dict]]
    data_quality: dict[str, object]


def extract_price_features(price_history: pd.DataFrame) -> dict[str, float | None]:
    if price_history.empty or "close" not in price_history:
        return {}

    prices = price_history.sort_values("date").copy()
    close = prices["close"].astype(float)
    volume = prices.get("volume", pd.Series(dtype=float)).astype(float)

    def pct(periods: int) -> float | None:
        if len(close) <= periods or close.iloc[-periods] == 0:
            return None
        return float((close.iloc[-1] / close.iloc[-periods]) - 1)

    recent_volume = volume.tail(10).mean() if not volume.empty else None
    base_volume = volume.tail(60).mean() if len(volume) >= 20 else None
    volume_surge = None
    if recent_volume and base_volume and base_volume != 0:
        volume_surge = float(recent_volume / base_volume)

    daily_returns = close.pct_change().dropna()
    volatility = (
        float(daily_returns.tail(63).std() * (252**0.5)) if not daily_returns.empty else None
    )

    return {
        "relative_strength_3m": pct(63),
        "relative_strength_6m": pct(126),
        "volume_surge_ratio": volume_surge,
        "volatility": volatility,
    }


def risk_score(features: dict[str, float | None]) -> float:
    volatility = features.get("volatility")
    debt_to_ebitda = features.get("debt_to_ebitda")
    valuation_zscore = features.get("valuation_zscore")

    volatility_risk = clip((volatility or 0) / 0.8) * 45
    debt_risk = clip((debt_to_ebitda or 0) / 5) * 30
    valuation_risk = clip((valuation_zscore or 0) / 4) * 25
    return round(volatility_risk + debt_risk + valuation_risk, 2)


def recommendation(overall_score: float, risk: float) -> str:
    if overall_score >= 80 and risk <= 55:
        return "Strong Buy"
    if overall_score >= 65 and risk <= 65:
        return "Buy"
    if overall_score >= 45 and risk <= 80:
        return "Watchlist"
    return "Avoid"


def score_ticker(
    ticker: str,
    price_history: pd.DataFrame,
    manual_features: dict[str, float | None] | None = None,
    live_price_data_available: bool | None = None,
    fallback_data_used: bool = False,
) -> ScoreResult:
    validate_weights()
    price_features = extract_price_features(price_history)
    feature_set = build_feature_set(ticker, price_features, manual_features)
    features = feature_set.values

    short_score, short_components = score_model_a(features)
    medium_score, medium_components = score_model_b(features)
    long_score, long_components = score_model_c(features)

    narrative = round(clip(features.get("narrative_score") or 0, 0, 10), 2)
    capital_flow = round(clip(features.get("capital_flow_score") or 0, 0, 10), 2)
    surprise = round(clip(features.get("surprise_score") or 0, 0, 10), 2)
    risk = risk_score(features)

    model_blend = (short_score * 0.25) + (medium_score * 0.35) + (long_score * 0.40)
    auxiliary = ((narrative + capital_flow + surprise) / 30) * 10
    risk_penalty = risk * 0.08
    overall = round(clip(model_blend + auxiliary - risk_penalty, 0, 100), 2)
    missing = missing_metrics(features)
    if live_price_data_available is None:
        live_price_data_available = not price_history.empty and not fallback_data_used

    return ScoreResult(
        ticker=ticker.upper(),
        short_score=short_score,
        medium_score=medium_score,
        long_score=long_score,
        narrative_score=narrative,
        capital_flow_score=capital_flow,
        surprise_score=surprise,
        risk_score=risk,
        overall_score=overall,
        recommendation=recommendation(overall, risk),
        components={
            "short_term_alpha": [component.as_dict() for component in short_components],
            "medium_term_alpha": [component.as_dict() for component in medium_components],
            "long_term_compounder": [component.as_dict() for component in long_components],
        },
        data_quality={
            "live_price_data_available": live_price_data_available,
            "fundamental_data_available": feature_set.fundamental_data_available,
            "fallback_data_used": fallback_data_used,
            "sample_fundamentals_used": feature_set.sample_fundamentals_used,
            "missing_metrics": missing,
        },
    )
