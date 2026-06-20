from tae.connectors.fallback import sample_price_history
from tae.forecasting.alpha_validation import (
    ALPHA_HORIZONS,
    alpha_validation_frame,
    benchmark_comparison,
    benchmark_return_frame,
    confidence_framework,
    final_investment_outcome_card,
    investment_outcome_by_bucket,
    predictor_validation_metrics,
    score_bucket_performance,
    threshold_analysis,
)


def sample_universe_prices():
    return {
        ticker: sample_price_history(start="2015-01-01", end="2026-01-01")
        for ticker in ["AAPL", "MSFT", "META"]
    }


def test_alpha_validation_frame_records_full_research_observations():
    prices = sample_universe_prices()
    frame = alpha_validation_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )

    assert not frame.empty
    assert set(ALPHA_HORIZONS).issubset(set(frame["horizon"]))
    assert {
        "score",
        "ticker",
        "date",
        "horizon",
        "predicted_return",
        "actual_return",
        "win_loss",
        "drawdown",
        "volatility",
    }.issubset(frame.columns)


def test_score_bucket_validation_and_investment_simulation_work():
    prices = sample_universe_prices()
    frame = alpha_validation_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )

    bucket_frame = score_bucket_performance(frame)
    investment_frame = investment_outcome_by_bucket(bucket_frame, 10)
    metrics = predictor_validation_metrics(frame)

    assert not bucket_frame.empty
    assert {
        "observation_count",
        "average_return_pct",
        "median_return_pct",
        "win_rate_pct",
        "sharpe_ratio",
        "maximum_drawdown_pct",
    }.issubset(bucket_frame.columns)
    assert not investment_frame.empty
    assert "expected_value" in investment_frame.columns
    assert "predictive_power" in metrics


def test_threshold_benchmark_confidence_and_final_outcome_card_work():
    prices = sample_universe_prices()
    frame = alpha_validation_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )
    bucket_frame = score_bucket_performance(frame)
    benchmark = benchmark_return_frame(
        "S&P 500",
        sample_price_history(start="2015-01-01", end="2026-01-01"),
        start_date="2016-01-01",
        step_days=252,
    )

    threshold_frame = threshold_analysis(frame)
    benchmark_frame = benchmark_comparison(frame, {"S&P 500": benchmark})
    confidence = confidence_framework(bucket_frame, score=85, horizon="12 months")
    card = final_investment_outcome_card(bucket_frame, score=85, investment_amount=10)

    assert not threshold_frame.empty
    assert {"threshold", "opportunity_count", "cagr_pct"}.issubset(threshold_frame.columns)
    assert not benchmark_frame.empty
    assert {"benchmark", "alpha_pct", "excess_return_pct"}.issubset(
        benchmark_frame.columns
    )
    assert confidence["score_bucket"] == "80-100"
    assert not card.empty
    assert "expected_value" in card.columns
