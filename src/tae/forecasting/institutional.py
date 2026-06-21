from __future__ import annotations

import numpy as np
import pandas as pd

from tae.backtesting.metrics import max_drawdown, sharpe_ratio, win_rate
from tae.forecasting.alpha_validation import annualized_return, horizon_years

MASTER_RANK_WEIGHTS = {
    "overall_score": 0.30,
    "short_term_opportunity_score": 0.20,
    "confidence_pct": 0.20,
    "empirical_12m_return": 0.15,
    "empirical_5y_return": 0.15,
}


def normalize_to_100(value: float, floor: float = 0.0, ceiling: float = 100.0) -> float:
    if ceiling == floor:
        return 0.0
    return max(0.0, min(100.0, ((value - floor) / (ceiling - floor)) * 100))


def master_rank_score(
    overall_score: float,
    short_term_opportunity_score: float,
    confidence_pct: float,
    empirical_12m_return: float,
    empirical_5y_return: float,
) -> float:
    score = (
        normalize_to_100(overall_score) * MASTER_RANK_WEIGHTS["overall_score"]
        + normalize_to_100(short_term_opportunity_score)
        * MASTER_RANK_WEIGHTS["short_term_opportunity_score"]
        + normalize_to_100(confidence_pct) * MASTER_RANK_WEIGHTS["confidence_pct"]
        + normalize_to_100(empirical_12m_return, -20, 50)
        * MASTER_RANK_WEIGHTS["empirical_12m_return"]
        + normalize_to_100(empirical_5y_return, -50, 300)
        * MASTER_RANK_WEIGHTS["empirical_5y_return"]
    )
    return round(score, 2)


def add_master_rank(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["master_rank_score"] = ranked.apply(
        lambda row: master_rank_score(
            float(row.get("overall_score", row.get("score", 0))),
            float(row.get("short_term_opportunity_score", row.get("score", 0))),
            float(row.get("confidence_pct", 50)),
            float(row.get("empirical_12m_return", row.get("actual_return", 0) * 100)),
            float(row.get("empirical_5y_return", row.get("actual_return", 0) * 100)),
        ),
        axis=1,
    )
    return ranked


def institutional_validation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["overall_score"] = working.get("overall_score", working.get("score", 0))
    working["short_term_opportunity_score"] = working.get(
        "short_term_opportunity_score",
        working["overall_score"],
    )
    working["confidence_pct"] = working.get("confidence_pct", 50.0)
    working["empirical_12m_return"] = working.get("empirical_12m_return", working["actual_return"] * 100)
    working["empirical_5y_return"] = working.get("empirical_5y_return", working["actual_return"] * 100)
    return add_master_rank(working)


def top_decile_test(
    frame: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
    horizons: list[str] | None = None,
) -> pd.DataFrame:
    horizons = horizons or ["1 month", "3 months", "12 months"]
    working = institutional_validation_frame(frame)
    if working.empty:
        return pd.DataFrame()
    rows = []
    for horizon in horizons:
        horizon_frame = working[working["horizon"] == horizon]
        selected = top_decile_observations(horizon_frame)
        rows.append(validation_summary_row("Top Decile", horizon, selected, benchmark_frames))
    return pd.DataFrame(rows)


def top_decile_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = []
    for _date, group in frame.groupby("date", sort=False):
        count = max(1, int(np.ceil(len(group) * 0.10)))
        selected.append(group.sort_values("master_rank_score", ascending=False).head(count))
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def top_20_portfolio_test(
    frame: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
    horizon: str = "1 month",
) -> pd.DataFrame:
    working = institutional_validation_frame(frame)
    if working.empty:
        return pd.DataFrame()
    horizon_frame = working[working["horizon"] == horizon]
    returns = []
    for date, group in horizon_frame.groupby("date", sort=False):
        selected = group.sort_values("master_rank_score", ascending=False).head(20)
        returns.append({"date": date, "horizon": horizon, "actual_return": selected["actual_return"].mean()})
    portfolio = pd.DataFrame(returns)
    return pd.DataFrame([validation_summary_row("Top 20 Portfolio", horizon, portfolio, benchmark_frames)])


def validation_summary_row(
    strategy: str,
    horizon: str,
    selected: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None,
) -> dict[str, float | str | int]:
    returns = selected["actual_return"].astype(float) if not selected.empty else pd.Series(dtype=float)
    years = horizon_years(horizon)
    benchmark_alpha = benchmark_alpha_pct(returns, horizon, benchmark_frames)
    return {
        "strategy": strategy,
        "horizon": horizon,
        "observation_count": int(len(selected)),
        "average_return_pct": float(returns.mean() * 100) if not returns.empty else 0.0,
        "cagr_pct": annualized_return(float(returns.mean()) if not returns.empty else 0.0, years) * 100,
        "annual_volatility_pct": float(returns.std() * np.sqrt(252 / max(1, horizon_years(horizon) * 252)) * 100)
        if len(returns) > 1
        else 0.0,
        "win_rate_pct": win_rate(returns) * 100,
        "sharpe_ratio": sharpe_ratio(returns, annualization=12),
        "sortino_ratio": sortino_ratio(returns),
        "maximum_drawdown_pct": max_drawdown(returns) * 100,
        "alpha_vs_sp500_pct": benchmark_alpha.get("S&P 500", 0.0),
        "alpha_vs_nasdaq100_pct": benchmark_alpha.get("Nasdaq 100", 0.0),
        "alpha_pct": benchmark_alpha.get("S&P 500", 0.0),
    }


def benchmark_alpha_pct(
    strategy_returns: pd.Series,
    horizon: str,
    benchmark_frames: dict[str, pd.DataFrame] | None,
) -> dict[str, float]:
    if not benchmark_frames or strategy_returns.empty:
        return {"S&P 500": 0.0, "Nasdaq 100": 0.0}
    strategy_average = float(strategy_returns.mean())
    alpha = {}
    for name, benchmark in benchmark_frames.items():
        if benchmark.empty:
            alpha[name] = 0.0
            continue
        group = benchmark[benchmark["horizon"] == horizon]
        alpha[name] = 0.0 if group.empty else float((strategy_average - group["actual_return"].mean()) * 100)
    return alpha


def sortino_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    downside_std = downside.std()
    if returns.empty or downside_std == 0 or pd.isna(downside_std):
        return 0.0
    return float((returns.mean() / downside_std) * np.sqrt(12))


def alpha_consistency_score(strategy_frame: pd.DataFrame, benchmark_frame: pd.DataFrame | None = None) -> float:
    if strategy_frame.empty:
        return 0.0
    returns = strategy_frame["actual_return"].astype(float)
    if benchmark_frame is not None and not benchmark_frame.empty:
        merged = strategy_frame.merge(
            benchmark_frame[["date", "horizon", "actual_return"]].rename(
                columns={"actual_return": "benchmark_return"}
            ),
            on=["date", "horizon"],
            how="left",
        )
        outperformance = float((merged["actual_return"] > merged["benchmark_return"].fillna(0)).mean())
    else:
        outperformance = float((returns > 0).mean())
    stability = max(0.0, 1 - float(returns.std() or 0.0))
    persistence = float(returns.rolling(3, min_periods=1).mean().gt(0).mean())
    return round((outperformance * 45 + stability * 25 + persistence * 30), 2)


def alpha_consistency_label(score: float) -> str:
    if score >= 90:
        return "Exceptional"
    if score >= 75:
        return "Strong"
    if score >= 60:
        return "Moderate"
    return "Weak"


def regime_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = institutional_validation_frame(frame)
    working["regime"] = working["actual_return"].map(regime_label)
    rows = []
    for (regime, horizon), group in working.groupby(["regime", "horizon"], sort=False):
        returns = group["actual_return"].astype(float)
        rows.append(
            {
                "regime": regime,
                "horizon": horizon,
                "master_rank_average_return_pct": float(returns.mean() * 100),
                "opportunity_finder_win_rate_pct": win_rate(returns) * 100,
                "portfolio_builder_sharpe": sharpe_ratio(returns, annualization=12),
            }
        )
    return pd.DataFrame(rows)


def regime_label(return_value: float) -> str:
    if return_value >= 0.08:
        return "Bull Market"
    if return_value <= -0.08:
        return "Bear Market"
    return "Sideways Market"


def false_positive_analysis(frame: pd.DataFrame) -> dict[str, float | int | str]:
    if frame.empty:
        return {
            "signals": 0,
            "winners": 0,
            "losers": 0,
            "false_positive_rate_pct": 0.0,
            "average_drawdown_pct": 0.0,
            "failed_prediction_factor": "Insufficient evidence",
        }
    signal_mask = pd.Series([True] * len(frame), index=frame.index)
    if "label" in frame:
        signal_mask &= frame["label"].isin(["Strong Buy", "Buy"])
    if "short_term_opportunity_label" in frame:
        signal_mask |= frame["short_term_opportunity_label"].eq("Swing Buy Now")
    signals = frame[signal_mask]
    winners = int((signals["actual_return"] > 0).sum()) if "actual_return" in signals else 0
    losers = int((signals["actual_return"] <= 0).sum()) if "actual_return" in signals else 0
    total = winners + losers
    return {
        "signals": int(len(signals)),
        "winners": winners,
        "losers": losers,
        "false_positive_rate_pct": 0.0 if total == 0 else losers / total * 100,
        "average_drawdown_pct": float(signals.get("drawdown", pd.Series(dtype=float)).mean() * 100)
        if not signals.empty
        else 0.0,
        "failed_prediction_factor": failed_prediction_factor(signals),
    }


def failed_prediction_factor(signals: pd.DataFrame) -> str:
    if signals.empty or "actual_return" not in signals:
        return "Insufficient evidence"
    losers = signals[signals["actual_return"] <= 0]
    if losers.empty:
        return "No dominant failed factor"
    factor_columns = [
        "master_rank_score",
        "short_term_opportunity_score",
        "confidence_pct",
        "empirical_12m_return",
        "empirical_5y_return",
    ]
    available = [column for column in factor_columns if column in losers]
    if not available:
        return "Actual return weakness"
    return str(losers[available].mean().idxmax())


def quality_of_edge_metrics(
    validation_frame: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float | str]:
    working = institutional_validation_frame(validation_frame)
    top_decile = top_decile_test(working, benchmark_frames)
    top_20 = top_20_portfolio_test(working, benchmark_frames)
    consistency = alpha_consistency_score(top_decile_observations(working), next(iter(benchmark_frames.values())) if benchmark_frames else None)
    errors = working["actual_return"] - working.get("predicted_return", working.get("forecast_return", 0))
    correlation = safe_corr(working.get("master_rank_score", pd.Series(dtype=float)), working["actual_return"]) if not working.empty else 0.0
    calibration_accuracy = max(0.0, 100 - float(errors.abs().mean() * 100)) if not working.empty else 0.0
    false_positive = false_positive_analysis(working)
    verdict = quality_of_edge_verdict(
        consistency,
        float(top_decile["alpha_pct"].mean()) if not top_decile.empty else 0.0,
        correlation,
        calibration_accuracy,
        float(false_positive["false_positive_rate_pct"]),
    )
    return {
        "master_rank_average_return_pct": float(working["actual_return"].mean() * 100) if not working.empty else 0.0,
        "top_decile_alpha_pct": float(top_decile["alpha_pct"].mean()) if not top_decile.empty else 0.0,
        "top_20_alpha_pct": float(top_20["alpha_pct"].mean()) if not top_20.empty else 0.0,
        "alpha_consistency_score": consistency,
        "alpha_consistency_label": alpha_consistency_label(consistency),
        "false_positive_rate_pct": float(false_positive["false_positive_rate_pct"]),
        "forecast_actual_correlation": correlation,
        "calibration_accuracy_pct": calibration_accuracy,
        "final_verdict": verdict,
    }


def quality_of_edge_verdict(
    consistency: float,
    alpha_pct: float,
    correlation: float,
    calibration_accuracy_pct: float,
    false_positive_rate_pct: float,
) -> str:
    if consistency >= 90 and alpha_pct > 5 and correlation > 0.25 and calibration_accuracy_pct > 75:
        return "Exceptional Edge"
    if consistency >= 75 and alpha_pct > 2 and correlation > 0.15 and false_positive_rate_pct < 45:
        return "Strong Edge"
    if consistency >= 60 and alpha_pct > 0 and correlation > 0.05:
        return "Moderate Edge"
    if consistency >= 45:
        return "Weak Edge"
    return "No Demonstrated Edge"


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty or left.std() == 0 or right.std() == 0:
        return 0.0
    correlation = left.corr(right)
    if pd.isna(correlation):
        return 0.0
    return float(correlation)
