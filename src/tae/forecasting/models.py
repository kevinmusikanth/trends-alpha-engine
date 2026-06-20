from __future__ import annotations

from dataclasses import dataclass


TRADING_DAYS = {
    "1 week": 5,
    "1 month": 21,
    "3 months": 63,
    "6 months": 126,
    "12 months": 252,
}

HORIZON_GROUPS = {
    "Short-term trading forecast": ["1 week", "1 month"],
    "Medium-term alpha forecast": ["3 months", "6 months"],
    "Long-term compounder forecast": ["12 months", "3 year CAGR", "5 year CAGR"],
}

FACTOR_LABELS = {
    "momentum": "Momentum",
    "volume": "Volume surge",
    "relative_strength": "Relative strength",
    "sector_momentum": "Sector momentum",
    "growth": "Growth",
    "valuation": "Valuation reasonableness",
    "quality": "Quality",
    "institutional": "Institutional/analyst support",
    "risk": "Volatility/risk",
}

# Conservative factor-to-forward-return priors. In a later calibration step these
# constants should be replaced by fitted coefficients from saved backtests.
HORIZON_FACTOR_COEFFICIENTS = {
    "1 week": {
        "momentum": 0.020,
        "volume": 0.012,
        "relative_strength": 0.018,
        "sector_momentum": 0.008,
        "growth": 0.004,
        "valuation": 0.002,
        "quality": 0.002,
        "institutional": 0.008,
        "risk": -0.012,
    },
    "1 month": {
        "momentum": 0.045,
        "volume": 0.020,
        "relative_strength": 0.035,
        "sector_momentum": 0.018,
        "growth": 0.015,
        "valuation": 0.008,
        "quality": 0.006,
        "institutional": 0.018,
        "risk": -0.025,
    },
    "3 months": {
        "momentum": 0.060,
        "volume": 0.015,
        "relative_strength": 0.050,
        "sector_momentum": 0.030,
        "growth": 0.050,
        "valuation": 0.025,
        "quality": 0.025,
        "institutional": 0.030,
        "risk": -0.045,
    },
    "6 months": {
        "momentum": 0.070,
        "volume": 0.010,
        "relative_strength": 0.055,
        "sector_momentum": 0.040,
        "growth": 0.080,
        "valuation": 0.045,
        "quality": 0.050,
        "institutional": 0.035,
        "risk": -0.065,
    },
    "12 months": {
        "momentum": 0.060,
        "volume": 0.004,
        "relative_strength": 0.050,
        "sector_momentum": 0.050,
        "growth": 0.130,
        "valuation": 0.080,
        "quality": 0.100,
        "institutional": 0.040,
        "risk": -0.090,
    },
}


@dataclass(frozen=True)
class ForecastLine:
    group: str
    horizon: str
    expected_return_pct: float
    bear_case_pct: float
    base_case_pct: float
    bull_case_pct: float
    confidence_pct: float


@dataclass(frozen=True)
class ValuationEstimate:
    current_price: float
    estimated_fair_value: float
    upside_downside_pct: float
    probability_positive_return_pct: float
    risk_rating: str
    suggested_label: str


@dataclass(frozen=True)
class ForecastDriver:
    factor: str
    contribution_pct: float
    exposure_pct: float


@dataclass(frozen=True)
class ForecastReport:
    ticker: str
    forecasts: list[ForecastLine]
    valuation: ValuationEstimate
    top_positive_drivers: list[ForecastDriver]
    top_negative_drivers: list[ForecastDriver]
    factor_exposures: dict[str, float]
    data_quality: dict[str, object]

