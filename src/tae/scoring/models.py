from __future__ import annotations

from tae.scoring.components import ComponentScore, score_inverse, score_positive


MODEL_A_WEIGHTS = {
    "sector_momentum": 10,
    "relative_strength": 15,
    "volume_surge": 15,
    "positive_news_flow": 15,
    "earnings_surprise": 15,
    "analyst_revisions": 10,
    "institutional_accumulation": 10,
    "short_squeeze_potential": 10,
}

MODEL_B_WEIGHTS = {
    "revenue_growth": 15,
    "eps_growth": 15,
    "margin_expansion": 10,
    "relative_strength": 15,
    "sector_momentum": 10,
    "guidance_upgrades": 15,
    "institutional_buying": 10,
    "valuation_reasonableness": 10,
}

MODEL_C_WEIGHTS = {
    "revenue_cagr": 15,
    "eps_cagr": 15,
    "roic": 15,
    "free_cash_flow_growth": 15,
    "balance_sheet_quality": 10,
    "competitive_moat": 10,
    "management_quality": 10,
    "industry_tailwinds": 10,
}


def validate_weights() -> None:
    for name, weights in {
        "Model A": MODEL_A_WEIGHTS,
        "Model B": MODEL_B_WEIGHTS,
        "Model C": MODEL_C_WEIGHTS,
    }.items():
        total = sum(weights.values())
        if total != 100:
            raise ValueError(f"{name} weights must sum to 100, got {total}")


def positive_component(
    label: str,
    feature_key: str,
    target: float,
    weight: float,
    features: dict[str, float | None],
) -> ComponentScore:
    return ComponentScore(label, weight, score_positive(features.get(feature_key), target, weight))


def inverse_component(
    label: str,
    feature_key: str,
    target: float,
    weight: float,
    features: dict[str, float | None],
) -> ComponentScore:
    return ComponentScore(label, weight, score_inverse(features.get(feature_key), target, weight))


def score_model_a(features: dict[str, float | None]) -> tuple[float, list[ComponentScore]]:
    weights = MODEL_A_WEIGHTS
    components = [
        positive_component(
            "Sector Momentum", "sector_momentum_3m", 0.12, weights["sector_momentum"], features
        ),
        positive_component(
            "Relative Strength",
            "relative_strength_3m",
            0.15,
            weights["relative_strength"],
            features,
        ),
        positive_component(
            "Volume Surge", "volume_surge_ratio", 2.0, weights["volume_surge"], features
        ),
        positive_component(
            "Positive News Flow", "news_sentiment", 1.0, weights["positive_news_flow"], features
        ),
        positive_component(
            "Earnings Surprise", "earnings_surprise", 0.15, weights["earnings_surprise"], features
        ),
        positive_component(
            "Analyst Revisions",
            "analyst_revision_ratio",
            1.0,
            weights["analyst_revisions"],
            features,
        ),
        positive_component(
            "Institutional Accumulation",
            "institutional_accumulation",
            1.0,
            weights["institutional_accumulation"],
            features,
        ),
        positive_component(
            "Short Squeeze Potential",
            "short_squeeze_potential",
            1.0,
            weights["short_squeeze_potential"],
            features,
        ),
    ]
    return round(sum(component.score for component in components), 2), components


def score_model_b(features: dict[str, float | None]) -> tuple[float, list[ComponentScore]]:
    weights = MODEL_B_WEIGHTS
    components = [
        positive_component(
            "Revenue Growth", "revenue_growth_yoy", 0.25, weights["revenue_growth"], features
        ),
        positive_component("EPS Growth", "eps_growth_yoy", 0.25, weights["eps_growth"], features),
        positive_component(
            "Margin Expansion", "margin_expansion", 0.05, weights["margin_expansion"], features
        ),
        positive_component(
            "Relative Strength", "relative_strength_6m", 0.2, weights["relative_strength"], features
        ),
        positive_component(
            "Sector Momentum", "sector_momentum_6m", 0.15, weights["sector_momentum"], features
        ),
        positive_component(
            "Guidance Upgrades", "guidance_upgrades", 1.0, weights["guidance_upgrades"], features
        ),
        positive_component(
            "Institutional Buying",
            "institutional_buying",
            1.0,
            weights["institutional_buying"],
            features,
        ),
        inverse_component(
            "Valuation Reasonableness",
            "valuation_zscore",
            3.0,
            weights["valuation_reasonableness"],
            features,
        ),
    ]
    return round(sum(component.score for component in components), 2), components


def score_model_c(features: dict[str, float | None]) -> tuple[float, list[ComponentScore]]:
    weights = MODEL_C_WEIGHTS
    components = [
        positive_component(
            "Revenue CAGR", "revenue_cagr_3y", 0.2, weights["revenue_cagr"], features
        ),
        positive_component("EPS CAGR", "eps_cagr_3y", 0.2, weights["eps_cagr"], features),
        positive_component("ROIC", "roic", 0.18, weights["roic"], features),
        positive_component(
            "Free Cash Flow Growth",
            "fcf_growth_3y",
            0.2,
            weights["free_cash_flow_growth"],
            features,
        ),
        inverse_component(
            "Balance Sheet Quality",
            "debt_to_ebitda",
            4.0,
            weights["balance_sheet_quality"],
            features,
        ),
        positive_component(
            "Competitive Moat", "competitive_moat", 1.0, weights["competitive_moat"], features
        ),
        positive_component(
            "Management Quality", "management_quality", 1.0, weights["management_quality"], features
        ),
        positive_component(
            "Industry Tailwinds", "industry_tailwinds", 1.0, weights["industry_tailwinds"], features
        ),
    ]
    return round(sum(component.score for component in components), 2), components
