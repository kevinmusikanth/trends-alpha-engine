from __future__ import annotations

from dataclasses import dataclass


FUNDAMENTAL_FEATURE_KEYS = {
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "margin_expansion",
    "guidance_upgrades",
    "institutional_buying",
    "valuation_zscore",
    "revenue_cagr_3y",
    "eps_cagr_3y",
    "roic",
    "fcf_growth_3y",
    "debt_to_ebitda",
    "competitive_moat",
    "management_quality",
    "industry_tailwinds",
}

SIGNAL_FEATURE_KEYS = {
    "sector_momentum_3m",
    "sector_momentum_6m",
    "news_sentiment",
    "earnings_surprise",
    "analyst_revision_ratio",
    "institutional_accumulation",
    "short_squeeze_potential",
    "narrative_score",
    "capital_flow_score",
    "surprise_score",
}

REQUIRED_FEATURE_KEYS = FUNDAMENTAL_FEATURE_KEYS | SIGNAL_FEATURE_KEYS | {
    "relative_strength_3m",
    "relative_strength_6m",
    "volume_surge_ratio",
    "volatility",
}


@dataclass(frozen=True)
class FeatureSet:
    values: dict[str, float | None]
    sample_fundamentals_used: bool
    fundamental_data_available: bool


SAMPLE_FUNDAMENTALS: dict[str, dict[str, float]] = {
    "AAPL": {
        "sector_momentum_3m": 0.06,
        "sector_momentum_6m": 0.10,
        "news_sentiment": 0.55,
        "earnings_surprise": 0.05,
        "analyst_revision_ratio": 0.45,
        "institutional_accumulation": 0.55,
        "short_squeeze_potential": 0.10,
        "revenue_growth_yoy": 0.05,
        "eps_growth_yoy": 0.10,
        "margin_expansion": 0.02,
        "guidance_upgrades": 0.40,
        "institutional_buying": 0.55,
        "valuation_zscore": 1.6,
        "revenue_cagr_3y": 0.07,
        "eps_cagr_3y": 0.12,
        "roic": 0.55,
        "fcf_growth_3y": 0.09,
        "debt_to_ebitda": 1.2,
        "competitive_moat": 0.95,
        "management_quality": 0.90,
        "industry_tailwinds": 0.70,
        "narrative_score": 6.5,
        "capital_flow_score": 6.0,
        "surprise_score": 5.0,
    },
    "MSFT": {
        "sector_momentum_3m": 0.09,
        "sector_momentum_6m": 0.16,
        "news_sentiment": 0.75,
        "earnings_surprise": 0.08,
        "analyst_revision_ratio": 0.65,
        "institutional_accumulation": 0.70,
        "short_squeeze_potential": 0.05,
        "revenue_growth_yoy": 0.15,
        "eps_growth_yoy": 0.18,
        "margin_expansion": 0.03,
        "guidance_upgrades": 0.65,
        "institutional_buying": 0.70,
        "valuation_zscore": 1.4,
        "revenue_cagr_3y": 0.15,
        "eps_cagr_3y": 0.18,
        "roic": 0.30,
        "fcf_growth_3y": 0.16,
        "debt_to_ebitda": 0.7,
        "competitive_moat": 0.95,
        "management_quality": 0.95,
        "industry_tailwinds": 0.90,
        "narrative_score": 8.5,
        "capital_flow_score": 7.5,
        "surprise_score": 6.5,
    },
    "NVDA": {
        "sector_momentum_3m": 0.14,
        "sector_momentum_6m": 0.28,
        "news_sentiment": 0.90,
        "earnings_surprise": 0.18,
        "analyst_revision_ratio": 0.85,
        "institutional_accumulation": 0.80,
        "short_squeeze_potential": 0.15,
        "revenue_growth_yoy": 0.75,
        "eps_growth_yoy": 0.95,
        "margin_expansion": 0.12,
        "guidance_upgrades": 0.90,
        "institutional_buying": 0.80,
        "valuation_zscore": 2.4,
        "revenue_cagr_3y": 0.55,
        "eps_cagr_3y": 0.70,
        "roic": 0.45,
        "fcf_growth_3y": 0.65,
        "debt_to_ebitda": 0.4,
        "competitive_moat": 0.90,
        "management_quality": 0.85,
        "industry_tailwinds": 0.95,
        "narrative_score": 10.0,
        "capital_flow_score": 8.5,
        "surprise_score": 9.0,
    },
    "AMZN": {
        "sector_momentum_3m": 0.08,
        "sector_momentum_6m": 0.15,
        "news_sentiment": 0.70,
        "earnings_surprise": 0.10,
        "analyst_revision_ratio": 0.70,
        "institutional_accumulation": 0.65,
        "short_squeeze_potential": 0.08,
        "revenue_growth_yoy": 0.11,
        "eps_growth_yoy": 0.45,
        "margin_expansion": 0.04,
        "guidance_upgrades": 0.70,
        "institutional_buying": 0.65,
        "valuation_zscore": 1.8,
        "revenue_cagr_3y": 0.13,
        "eps_cagr_3y": 0.35,
        "roic": 0.14,
        "fcf_growth_3y": 0.38,
        "debt_to_ebitda": 0.9,
        "competitive_moat": 0.90,
        "management_quality": 0.85,
        "industry_tailwinds": 0.85,
        "narrative_score": 8.0,
        "capital_flow_score": 7.0,
        "surprise_score": 7.0,
    },
    "PLTR": {
        "sector_momentum_3m": 0.13,
        "sector_momentum_6m": 0.25,
        "news_sentiment": 0.80,
        "earnings_surprise": 0.12,
        "analyst_revision_ratio": 0.70,
        "institutional_accumulation": 0.60,
        "short_squeeze_potential": 0.35,
        "revenue_growth_yoy": 0.22,
        "eps_growth_yoy": 0.35,
        "margin_expansion": 0.06,
        "guidance_upgrades": 0.75,
        "institutional_buying": 0.60,
        "valuation_zscore": 2.6,
        "revenue_cagr_3y": 0.20,
        "eps_cagr_3y": 0.40,
        "roic": 0.12,
        "fcf_growth_3y": 0.35,
        "debt_to_ebitda": 0.1,
        "competitive_moat": 0.75,
        "management_quality": 0.75,
        "industry_tailwinds": 0.90,
        "narrative_score": 9.0,
        "capital_flow_score": 7.5,
        "surprise_score": 7.5,
    },
}


def build_feature_set(
    ticker: str,
    price_features: dict[str, float | None],
    manual_features: dict[str, float | None] | None = None,
) -> FeatureSet:
    values = dict(price_features)
    supplied = manual_features or {}
    values.update(supplied)

    has_supplied_fundamentals = any(
        values.get(feature) is not None for feature in FUNDAMENTAL_FEATURE_KEYS
    )
    sample = SAMPLE_FUNDAMENTALS.get(ticker.upper())
    sample_used = False

    if not has_supplied_fundamentals and sample:
        values.update(sample)
        sample_used = True

    fundamental_data_available = any(
        values.get(feature) is not None for feature in FUNDAMENTAL_FEATURE_KEYS
    )
    return FeatureSet(
        values=values,
        sample_fundamentals_used=sample_used,
        fundamental_data_available=fundamental_data_available,
    )


def missing_metrics(features: dict[str, float | None]) -> list[str]:
    return sorted(
        feature for feature in REQUIRED_FEATURE_KEYS if features.get(feature) is None
    )

