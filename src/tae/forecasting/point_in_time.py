from __future__ import annotations

import numpy as np
import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecasting.alpha_validation import (
    ALPHA_HORIZONS,
    actual_forward_return_by_days,
    annualized_return,
    benchmark_return_frame,
    forward_window_risk,
    horizon_years,
    predicted_return_for_horizon,
)
from tae.forecasting.backtest import normalize_price_dates, normalize_timestamp
from tae.forecasting.engine import build_forecast_report
from tae.forecasting.universe import score_bucket_20_point
from tae.scoring.engine import score_ticker

FORECAST_CALIBRATION_BUCKETS = [
    (-np.inf, 0.04999, "0-5%"),
    (0.05, 0.09999, "5-10%"),
    (0.10, 0.14999, "10-15%"),
    (0.15, 0.19999, "15-20%"),
    (0.20, np.inf, "20%+"),
]


def point_in_time_prediction_frame(
    price_history_by_ticker: dict[str, pd.DataFrame],
    start_date: str,
    fallback_data_used_by_ticker: dict[str, bool] | None = None,
    horizons: list[str] | None = None,
    step_days: int = 21,
) -> pd.DataFrame:
    horizons = horizons or ALPHA_HORIZONS
    fallback_data_used_by_ticker = fallback_data_used_by_ticker or {}
    rows = []

    for ticker, price_history in price_history_by_ticker.items():
        prices = normalize_price_dates(price_history)
        if prices.empty:
            continue
        start = normalize_timestamp(start_date)
        eligible = prices.index[prices["date"] >= start]
        is_fallback = bool(fallback_data_used_by_ticker.get(ticker, False))

        for index in eligible[:: max(1, step_days)]:
            if index < 126:
                continue
            historical_prices = prices.iloc[: index + 1].copy()
            score = score_ticker(
                ticker,
                historical_prices,
                manual_features=point_in_time_proxy_features(historical_prices),
                live_price_data_available=not is_fallback,
                fallback_data_used=is_fallback,
                use_sample_fundamentals=False,
            )
            report = build_forecast_report(score, historical_prices)
            as_of_date = prices.loc[index, "date"]

            for horizon in horizons:
                actual = actual_forward_return_by_days(prices, int(index), horizon)
                if actual is None:
                    continue
                forecast = predicted_return_for_horizon(report, horizon)
                drawdown, volatility = forward_window_risk(prices, int(index), horizon)
                error = actual - forecast
                rows.append(
                    {
                        "date": as_of_date,
                        "ticker": ticker,
                        "score": score.overall_score,
                        "score_bucket": score_bucket_20_point(score.overall_score),
                        "horizon": horizon,
                        "forecast_return": forecast,
                        "actual_future_return": actual,
                        "prediction_error": error,
                        "absolute_prediction_error": abs(error),
                        "win_loss": "Win" if actual > 0 else "Loss",
                        "hit": bool(actual > 0),
                        "drawdown": drawdown,
                        "volatility": volatility,
                        "sample_fundamentals_used": score.data_quality.get(
                            "sample_fundamentals_used",
                            False,
                        ),
                        "fundamental_data_available": score.data_quality.get(
                            "fundamental_data_available",
                            False,
                        ),
                        "point_in_time_proxy_features_used": True,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "score",
                "score_bucket",
                "horizon",
                "forecast_return",
                "actual_future_return",
                "prediction_error",
                "absolute_prediction_error",
                "win_loss",
                "hit",
                "drawdown",
                "volatility",
                "sample_fundamentals_used",
                "fundamental_data_available",
                "point_in_time_proxy_features_used",
            ]
        )
    return pd.DataFrame(rows)


def point_in_time_proxy_features(historical_prices: pd.DataFrame) -> dict[str, float]:
    prices = normalize_price_dates(historical_prices)
    close = prices["close"].astype(float)
    volume = prices.get("volume", pd.Series(dtype=float)).astype(float)
    rs_3m = trailing_return(close, 63)
    rs_6m = trailing_return(close, 126)
    daily_returns = close.pct_change().dropna()
    volatility = float(daily_returns.tail(63).std() * np.sqrt(252)) if not daily_returns.empty else 0.25
    recent_volume = volume.tail(10).mean() if not volume.empty else 1.0
    base_volume = volume.tail(60).mean() if len(volume) >= 20 else recent_volume
    volume_surge = float(recent_volume / base_volume) if base_volume else 1.0
    momentum_quality = max(0.0, min(1.0, 0.5 + rs_6m * 2.0))

    return {
        "sector_momentum_3m": max(0.0, rs_3m * 0.7),
        "sector_momentum_6m": max(0.0, rs_6m * 0.7),
        "news_sentiment": max(0.0, min(1.0, 0.5 + rs_3m * 2.0)),
        "earnings_surprise": max(0.0, min(0.20, rs_3m * 0.6)),
        "analyst_revision_ratio": max(0.0, min(1.0, 0.5 + rs_6m * 2.0)),
        "institutional_accumulation": max(0.0, min(1.0, 0.45 + (volume_surge - 1) * 2)),
        "short_squeeze_potential": max(0.0, min(1.0, volatility / 1.2)),
        "revenue_growth_yoy": max(0.0, min(0.35, rs_6m * 0.8)),
        "eps_growth_yoy": max(0.0, min(0.45, rs_6m)),
        "margin_expansion": max(0.0, min(0.08, rs_3m * 0.18)),
        "guidance_upgrades": momentum_quality,
        "institutional_buying": max(0.0, min(1.0, 0.45 + (volume_surge - 1) * 2)),
        "valuation_zscore": max(0.3, min(3.0, 1.6 - rs_6m + volatility)),
        "revenue_cagr_3y": max(0.0, min(0.30, rs_6m * 0.65)),
        "eps_cagr_3y": max(0.0, min(0.35, rs_6m * 0.75)),
        "roic": max(0.05, min(0.30, 0.12 + momentum_quality * 0.12)),
        "fcf_growth_3y": max(0.0, min(0.30, rs_6m * 0.65)),
        "debt_to_ebitda": max(0.2, min(3.5, 1.4 + volatility - rs_6m)),
        "competitive_moat": max(0.35, min(0.90, 0.45 + momentum_quality * 0.35)),
        "management_quality": max(0.35, min(0.90, 0.45 + momentum_quality * 0.30)),
        "industry_tailwinds": max(0.35, min(0.90, 0.45 + momentum_quality * 0.35)),
        "narrative_score": max(0.0, min(10.0, 5.0 + rs_6m * 18)),
        "capital_flow_score": max(0.0, min(10.0, 5.0 + (volume_surge - 1) * 12)),
        "surprise_score": max(0.0, min(10.0, 5.0 + rs_3m * 18)),
    }


def trailing_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods or close.iloc[-periods] == 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-periods] - 1)


def prediction_accuracy_metrics(frame: pd.DataFrame) -> dict[str, float | str]:
    if frame.empty:
        return {
            "average_prediction_error_pct": 0.0,
            "median_prediction_error_pct": 0.0,
            "rmse_pct": 0.0,
            "forecast_actual_correlation": 0.0,
            "calibration_accuracy_pct": 0.0,
        }
    errors = frame["prediction_error"].astype(float)
    forecast = frame["forecast_return"].astype(float)
    actual = frame["actual_future_return"].astype(float)
    return {
        "average_prediction_error_pct": float(errors.mean() * 100),
        "median_prediction_error_pct": float(errors.median() * 100),
        "rmse_pct": float(np.sqrt((errors**2).mean()) * 100),
        "forecast_actual_correlation": safe_corr(forecast, actual),
        "calibration_accuracy_pct": max(0.0, 100 - float(errors.abs().mean() * 100)),
    }


def score_threshold_validation(
    frame: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or [60, 70, 80, 90]
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for threshold in thresholds:
        eligible = frame[frame["score"] > threshold]
        for horizon in ALPHA_HORIZONS:
            group = eligible[eligible["horizon"] == horizon]
            if group.empty:
                rows.append(
                    {
                        "threshold": f"Score > {threshold}",
                        "horizon": horizon,
                        "observation_count": 0,
                        "average_actual_return_pct": 0.0,
                        "win_rate_pct": 0.0,
                        "sharpe_ratio": 0.0,
                        "maximum_drawdown_pct": 0.0,
                    }
                )
                continue
            returns = group["actual_future_return"].astype(float)
            rows.append(
                {
                    "threshold": f"Score > {threshold}",
                    "horizon": horizon,
                    "observation_count": int(len(group)),
                    "average_actual_return_pct": float(returns.mean() * 100),
                    "win_rate_pct": float(win_rate(returns) * 100),
                    "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                    "maximum_drawdown_pct": float(group["drawdown"].min() * 100),
                }
            )
    return pd.DataFrame(rows)


def investment_outcome_validation(
    frame: pd.DataFrame,
    investment_amount: float,
    benchmarks: dict[str, pd.DataFrame] | None = None,
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or [60, 70, 80, 90]
    target_horizons = ["12 months", "3 years", "5 years"]
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for threshold in thresholds:
        eligible = frame[frame["score"] > threshold]
        for horizon in target_horizons:
            group = eligible[eligible["horizon"] == horizon]
            if group.empty:
                row = {
                    "threshold": f"Score > {threshold}",
                    "horizon": horizon,
                    "observation_count": 0,
                    "total_invested": 0.0,
                    "portfolio_value": 0.0,
                    "cagr_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "maximum_drawdown_pct": 0.0,
                }
                if benchmarks:
                    for name in benchmarks:
                        row[f"{name}_portfolio_value"] = 0.0
                        row[f"{name}_excess_value"] = 0.0
                rows.append(row)
                continue
            returns = group["actual_future_return"].astype(float)
            total_invested = investment_amount * len(group)
            portfolio_value = float((investment_amount * (1 + returns)).sum())
            years = horizon_years(horizon)
            cagr = annualized_return(portfolio_value / total_invested - 1, years)
            row = {
                "threshold": f"Score > {threshold}",
                "horizon": horizon,
                "observation_count": int(len(group)),
                "total_invested": total_invested,
                "portfolio_value": portfolio_value,
                "cagr_pct": cagr * 100,
                "sharpe_ratio": sharpe_ratio(returns, annualization=12),
                "maximum_drawdown_pct": max_drawdown(returns) * 100,
            }
            if benchmarks:
                for name, benchmark in benchmarks.items():
                    benchmark_returns = benchmark[benchmark["horizon"] == horizon][
                        "actual_return"
                    ].astype(float)
                    if benchmark_returns.empty:
                        continue
                    benchmark_value = total_invested * (1 + benchmark_returns.mean())
                    row[f"{name}_portfolio_value"] = float(benchmark_value)
                    row[f"{name}_excess_value"] = float(portfolio_value - benchmark_value)
            rows.append(row)
    return pd.DataFrame(rows)


def forecast_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    working = frame.copy()
    working["forecast_bucket"] = working["forecast_return"].map(forecast_bucket)
    for (horizon, bucket), group in working.groupby(["horizon", "forecast_bucket"], sort=False):
        rows.append(
            {
                "horizon": horizon,
                "forecast_bucket": bucket,
                "observation_count": int(len(group)),
                "average_forecast_return_pct": float(group["forecast_return"].mean() * 100),
                "average_actual_return_pct": float(
                    group["actual_future_return"].mean() * 100
                ),
                "average_prediction_error_pct": float(group["prediction_error"].mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def forecast_bucket(value: float) -> str:
    for low, high, label in FORECAST_CALIBRATION_BUCKETS:
        if low <= value <= high:
            return label
    return "0-5%"


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty or left.std() == 0 or right.std() == 0:
        return 0.0
    correlation = left.corr(right)
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


__all__ = [
    "benchmark_return_frame",
    "forecast_calibration",
    "investment_outcome_validation",
    "point_in_time_prediction_frame",
    "prediction_accuracy_metrics",
    "score_threshold_validation",
]
