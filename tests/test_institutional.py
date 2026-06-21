import pandas as pd

from tae.forecasting.institutional import (
    add_master_rank,
    alpha_consistency_label,
    alpha_consistency_score,
    false_positive_analysis,
    master_rank_score,
    quality_of_edge_metrics,
    top_20_portfolio_test,
    top_decile_test,
)


def sample_validation_frame():
    rows = []
    for date in pd.date_range("2021-01-01", periods=4, freq="MS"):
        for ticker_index in range(30):
            score = 100 - ticker_index
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_index}",
                    "horizon": "1 month",
                    "score": score,
                    "overall_score": score,
                    "short_term_opportunity_score": max(0, score - 5),
                    "confidence_pct": max(0, score - 10),
                    "empirical_12m_return": score / 2,
                    "empirical_5y_return": score * 2,
                    "actual_return": (score - 50) / 1000,
                    "predicted_return": (score - 55) / 1000,
                    "drawdown": -0.02,
                    "volatility": 0.20,
                    "label": "Strong Buy" if score > 90 else "Watchlist",
                    "short_term_opportunity_label": (
                        "Swing Buy Now" if score > 90 else "Watch"
                    ),
                }
            )
    return pd.DataFrame(rows)


def sample_benchmark():
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=4, freq="MS"),
            "horizon": ["1 month"] * 4,
            "actual_return": [0.01, 0.012, 0.008, 0.01],
            "drawdown": [-0.01] * 4,
            "volatility": [0.15] * 4,
        }
    )


def test_master_rank_score_normalizes_components():
    score = master_rank_score(
        overall_score=80,
        short_term_opportunity_score=75,
        confidence_pct=70,
        empirical_12m_return=25,
        empirical_5y_return=150,
    )

    assert 0 <= score <= 100
    assert score > 60


def test_top_decile_test_compares_against_benchmark():
    frame = sample_validation_frame()
    result = top_decile_test(
        frame,
        benchmark_frames={"S&P 500": sample_benchmark(), "Nasdaq 100": sample_benchmark()},
        horizons=["1 month"],
    )

    assert not result.empty
    assert {"cagr_pct", "win_rate_pct", "alpha_vs_sp500_pct"}.issubset(result.columns)
    assert result.iloc[0]["observation_count"] == 12


def test_top_20_portfolio_test_runs_monthly_simulation():
    frame = sample_validation_frame()
    result = top_20_portfolio_test(
        frame,
        benchmark_frames={"S&P 500": sample_benchmark(), "Nasdaq 100": sample_benchmark()},
    )

    assert not result.empty
    assert {"annual_volatility_pct", "sortino_ratio", "maximum_drawdown_pct"}.issubset(
        result.columns
    )


def test_alpha_consistency_and_quality_metrics():
    frame = add_master_rank(sample_validation_frame())
    benchmark = sample_benchmark()

    consistency = alpha_consistency_score(frame, benchmark)
    metrics = quality_of_edge_metrics(
        frame,
        benchmark_frames={"S&P 500": benchmark, "Nasdaq 100": benchmark},
    )
    false_positive = false_positive_analysis(frame)

    assert consistency > 0
    assert alpha_consistency_label(consistency) in {"Exceptional", "Strong", "Moderate", "Weak"}
    assert metrics["final_verdict"] in {
        "Exceptional Edge",
        "Strong Edge",
        "Moderate Edge",
        "Weak Edge",
        "No Demonstrated Edge",
    }
    assert "false_positive_rate_pct" in false_positive
