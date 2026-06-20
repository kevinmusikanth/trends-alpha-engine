from __future__ import annotations

import pandas as pd

from tae.forecasting.models import ForecastReport

OUTCOME_HORIZONS = [
    {"label": "1 Month", "forecast_horizon": "1 month", "years": 1 / 12, "is_cagr": False},
    {"label": "3 Months", "forecast_horizon": "3 months", "years": 3 / 12, "is_cagr": False},
    {"label": "12 Months", "forecast_horizon": "12 months", "years": 1.0, "is_cagr": False},
    {"label": "3 Years", "forecast_horizon": "3 year CAGR", "years": 3.0, "is_cagr": True},
    {"label": "5 Years", "forecast_horizon": "5 year CAGR", "years": 5.0, "is_cagr": True},
]


def investment_outcome_projection(report: ForecastReport, investment_amount: float) -> pd.DataFrame:
    forecasts = {line.horizon: line for line in report.forecasts}
    probability_positive = report.valuation.probability_positive_return_pct
    rows = []

    for horizon in OUTCOME_HORIZONS:
        forecast = forecasts[horizon["forecast_horizon"]]
        bear_value = projected_value(
            investment_amount,
            forecast.bear_case_pct,
            years=horizon["years"],
            is_cagr=horizon["is_cagr"],
        )
        base_value = projected_value(
            investment_amount,
            forecast.base_case_pct,
            years=horizon["years"],
            is_cagr=horizon["is_cagr"],
        )
        bull_value = projected_value(
            investment_amount,
            forecast.bull_case_pct,
            years=horizon["years"],
            is_cagr=horizon["is_cagr"],
        )
        rows.append(
            {
                "horizon": horizon["label"],
                "years": horizon["years"],
                "bear_case_return_pct": forecast.bear_case_pct,
                "base_case_return_pct": forecast.base_case_pct,
                "bull_case_return_pct": forecast.bull_case_pct,
                "bear_case_value": bear_value,
                "base_case_value": base_value,
                "bull_case_value": bull_value,
                "expected_value": base_value,
                "expected_cagr_pct": annualized_return(
                    investment_amount,
                    base_value,
                    horizon["years"],
                ),
                "probability_positive_return_pct": probability_positive,
                "probability_losing_money_pct": max(0.0, 100.0 - probability_positive),
            }
        )

    return pd.DataFrame(rows)


def projected_value(
    investment_amount: float,
    return_pct: float,
    years: float,
    is_cagr: bool,
) -> float:
    rate = return_pct / 100
    if is_cagr:
        return round(investment_amount * ((1 + rate) ** years), 2)
    return round(investment_amount * (1 + rate), 2)


def annualized_return(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0.0
    return round((((end_value / start_value) ** (1 / years)) - 1) * 100, 2)


def outcome_growth_paths(outcomes: pd.DataFrame, investment_amount: float) -> pd.DataFrame:
    rows = [
        {
            "horizon": "Today",
            "years": 0.0,
            "Bear": investment_amount,
            "Base": investment_amount,
            "Bull": investment_amount,
        }
    ]
    for row in outcomes.to_dict("records"):
        rows.append(
            {
                "horizon": row["horizon"],
                "years": row["years"],
                "Bear": row["bear_case_value"],
                "Base": row["base_case_value"],
                "Bull": row["bull_case_value"],
            }
        )
    return pd.DataFrame(rows)
