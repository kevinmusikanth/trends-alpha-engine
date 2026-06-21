import sys
from pathlib import Path
from types import SimpleNamespace


STREAMLIT_EMPIRICAL_HELPERS = (
    "current_bucket_return_distribution",
    "empirical_fallback_message",
    "empirical_investment_outcome_table",
    "empirical_outlook_interpretation",
    "empirical_score_bucket_forecast",
    "score_bucket_comparison",
)


def load_streamlit_app():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import streamlit_app

    return streamlit_app


def test_streamlit_app_imports_successfully():
    streamlit_app = load_streamlit_app()

    assert streamlit_app.ADVICE_WARNING == "Research tool only. Not financial advice."


def test_streamlit_empirical_helpers_import_successfully():
    from tae import forecasting
    from tae.forecasting import empirical

    for helper_name in STREAMLIT_EMPIRICAL_HELPERS:
        assert callable(getattr(empirical, helper_name))
        assert callable(getattr(forecasting, helper_name))


def test_safe_price_history_falls_back_when_yahoo_fails(monkeypatch):
    streamlit_app = load_streamlit_app()

    def raise_rate_limit(*args, **kwargs):
        raise RuntimeError("Too Many Requests")

    monkeypatch.setattr(streamlit_app, "load_price_history", raise_rate_limit)

    prices, is_fallback = streamlit_app.safe_price_history("AAPL", period="1y")

    assert is_fallback is True
    assert not prices.empty
    assert {"date", "close", "volume"}.issubset(prices.columns)


def test_streamlit_scoring_path_returns_quality_metadata():
    streamlit_app = load_streamlit_app()

    prices = streamlit_app.sample_price_history(periods=260)
    score, quality = streamlit_app.score_for_app("AAPL", prices, True)
    report, forecast_frame, valuation_frame, positive_frame, negative_frame, exposure_frame = (
        streamlit_app.forecast_frames(score, prices)
    )

    assert score.ticker == "AAPL"
    assert score.long_score > 0
    assert "fallback_data_used" in quality
    assert "sample_fundamentals_used" in quality
    assert "missing_metrics" in quality
    assert not forecast_frame.empty
    assert not valuation_frame.empty
    assert not positive_frame.empty
    assert not negative_frame.empty
    assert not exposure_frame.empty
    assert report.valuation.current_price > 0


def test_streamlit_scoring_path_handles_older_score_signature(monkeypatch):
    streamlit_app = load_streamlit_app()

    def old_score_ticker(ticker, prices):
        return SimpleNamespace(
            ticker=ticker,
            short_score=1,
            medium_score=2,
            long_score=3,
            risk_score=4,
            overall_score=5,
            recommendation="Watchlist",
            components={},
        )

    monkeypatch.setattr(streamlit_app, "score_ticker", old_score_ticker)
    prices = streamlit_app.sample_price_history(periods=5)
    score, quality = streamlit_app.score_for_app("AAPL", prices, True)

    assert score.ticker == "AAPL"
    assert quality["fallback_data_used"] is True
    assert quality["fundamental_data_available"] is False


def test_screener_ticker_input_splits_comma_separated_symbols():
    streamlit_app = load_streamlit_app()

    tickers = streamlit_app.custom_tickers_from_text("AAPL, MSFT, META, NVDA")

    assert tickers == ["AAPL", "MSFT", "META", "NVDA"]


def test_screener_row_includes_empirical_metrics():
    streamlit_app = load_streamlit_app()
    score = SimpleNamespace(
        ticker="AAPL",
        overall_score=82.5,
        recommendation="Buy",
        short_score=77.0,
        medium_score=83.0,
        long_score=88.0,
        risk_score=18.0,
    )
    empirical = streamlit_app.pd.DataFrame(
        [
            {
                "horizon": "1 week",
                "average_return_pct": 1.5,
                "win_rate_pct": 58.0,
                "observation_count": 300,
                "calibration_accuracy_pct": 80.0,
                "forecast_actual_correlation_pct": 75.0,
                "forecast_error_pct": 10.0,
            },
            {
                "horizon": "3 months",
                "average_return_pct": 18.0,
                "win_rate_pct": 76.0,
                "observation_count": 300,
                "calibration_accuracy_pct": 82.0,
                "forecast_actual_correlation_pct": 76.0,
                "forecast_error_pct": 9.0,
            },
            {
                "horizon": "12 months",
                "average_return_pct": 24.0,
                "win_rate_pct": 72.0,
                "observation_count": 300,
                "calibration_accuracy_pct": 80.0,
                "forecast_actual_correlation_pct": 75.0,
                "forecast_error_pct": 10.0,
            },
            {
                "horizon": "3 years",
                "average_return_pct": 90.0,
                "win_rate_pct": 81.0,
                "observation_count": 300,
                "calibration_accuracy_pct": 78.0,
                "forecast_actual_correlation_pct": 75.0,
                "forecast_error_pct": 10.0,
            },
            {
                "horizon": "5 years",
                "average_return_pct": 155.0,
                "win_rate_pct": 86.0,
                "observation_count": 300,
                "calibration_accuracy_pct": 76.0,
                "forecast_actual_correlation_pct": 75.0,
                "forecast_error_pct": 10.0,
            },
        ]
    )

    prices = streamlit_app.sample_price_history(periods=300)
    row = streamlit_app.screener_row_from_score(score, empirical, prices)

    assert row["ticker"] == "AAPL"
    assert row["empirical_12m_return"] == 24.0
    assert row["empirical_3y_return"] == 90.0
    assert row["empirical_5y_return"] == 155.0
    assert row["empirical_win_rate"] == 72.0
    assert row["best_holding_period"] == "5 Years"
    assert row["expected_value_12m"] == 12400.0
    assert row["expected_value_3y"] == 19000.0
    assert row["expected_value_5y"] == 25500.0
    assert row["best_expected_value"] == 25500.0
    assert row["empirical_outlook"] == "Strong Long-Term Edge"
    assert row["confidence_pct"] > 80
    assert 0 <= row["master_rank_score"] <= 100
    assert row["alpha_consistency_score"] > 0
    assert "momentum_explosion_score" in row
    assert "short_term_opportunity_score" in row
    assert "opportunity_horizon" in row
    assert "empirical_1w_return" in row
    assert "empirical_6w_return" in row
    assert "empirical_3m_return" in row
    assert "empirical_6m_return" in row
    assert "trading_score" in row
    assert "trading_action" in row
    assert "trading_horizon" in row
    assert "trading_expected_return" in row
    assert "trading_expected_annualized_return" in row
    assert "swing_score" in row
    assert "swing_action" in row
    assert "swing_horizon" in row
    assert "swing_expected_return" in row
    assert "swing_expected_annualized_return" in row
    assert "compounder_score" in row
    assert "compounder_action" in row
    assert "compounder_horizon" in row
    assert "compounder_expected_return" in row
    assert "compounder_expected_annualized_return" in row
    assert "risk_reward_ratio" in row
    assert "forecast_multiplier" in row
    assert "forecast_confidence_band" in row
    assert "expected_benchmark_return" in row
    assert "expected_alpha" in row
    assert row["advisory_score"] > 0
    assert "recommended_horizon_score" in row
    assert "risk_adjusted_return" in row
    assert "duration_penalty" in row
    assert row["advisory_action"] in {
        "Short-Term Opportunity",
        "Medium-Term Opportunity",
        "Long-Term Compounder",
        "Watchlist",
        "Avoid",
    }
    assert row["recommended_holding_period"] in streamlit_app.ADVISORY_HORIZONS
    assert row["historical_win_rate"] > 0
    assert row["confidence_level"] in {"High", "Good", "Moderate", "Low"}
    assert "Research indicates AAPL" in row["advisory_summary"]


def test_multi_horizon_advisory_fields_are_independent():
    streamlit_app = load_streamlit_app()
    base_row = {
        "empirical_1w_return": 3.0,
        "empirical_2w_return": 5.0,
        "empirical_4w_return": 4.0,
        "empirical_1w_win_rate": 60.0,
        "empirical_2w_win_rate": 62.0,
        "empirical_4w_win_rate": 58.0,
        "empirical_3m_return": 9.0,
        "empirical_6m_return": 14.0,
        "empirical_3m_win_rate": 63.0,
        "empirical_6m_win_rate": 66.0,
        "empirical_12m_return": 20.0,
        "empirical_3y_return": 80.0,
        "empirical_5y_return": 500.0,
        "empirical_win_rate": 70.0,
        "empirical_3y_win_rate": 82.0,
        "empirical_5y_win_rate": 88.0,
        "momentum_explosion_score": 90.0,
        "master_rank_score": 75.0,
        "alpha_consistency_score": 78.0,
        "confidence_pct": 72.0,
    }

    trading = streamlit_app.trading_advisory_fields(base_row)
    swing = streamlit_app.swing_advisory_fields(base_row)
    compounder = streamlit_app.compounder_advisory_fields(base_row)

    assert trading["trading_horizon"] == "2 weeks"
    assert trading["trading_expected_return"] == 5.0
    assert swing["swing_horizon"] == "6 months"
    assert swing["swing_expected_return"] == 14.0
    assert compounder["compounder_horizon"] == "5 years"
    assert compounder["compounder_expected_return"] == 500.0

    changed_long_term = {**base_row, "empirical_5y_return": 5_000.0}
    assert streamlit_app.trading_advisory_fields(changed_long_term) == trading
    assert streamlit_app.swing_advisory_fields(changed_long_term) == swing


def test_short_term_opportunity_labels_and_horizon_classification():
    streamlit_app = load_streamlit_app()

    assert streamlit_app.momentum_explosion_label(85) == "High Probability Runner"
    assert streamlit_app.short_term_opportunity_label(88) == "Swing Buy Now"
    assert streamlit_app.short_term_opportunity_label(72) == "Strong Momentum"
    assert streamlit_app.short_term_opportunity_label(56) == "Watch"
    assert streamlit_app.short_term_opportunity_label(40) == "Ignore"
    assert streamlit_app.opportunity_horizon_label(45, 80) == "Long-Term Investment"
    assert streamlit_app.opportunity_horizon_label(80, 45) == "Swing Trade"
    assert streamlit_app.opportunity_horizon_label(80, 80) == "Buy and Hold"
    assert streamlit_app.opportunity_horizon_label(45, 45) == "Avoid"


def test_momentum_explosion_score_has_expected_range():
    streamlit_app = load_streamlit_app()

    details = streamlit_app.momentum_explosion_details(
        streamlit_app.sample_price_history(periods=300)
    )

    assert 0 <= details["momentum_explosion_score"] <= 100
    assert details["momentum_explosion_label"] in {
        "High Probability Runner",
        "Positive Momentum",
        "Neutral",
        "Weak",
        "Avoid",
    }


def test_empirical_outlook_label_rules():
    streamlit_app = load_streamlit_app()

    assert streamlit_app.empirical_outlook_label(20, 300, 85) == (
        "Exceptional Long-Term Edge"
    )
    assert streamlit_app.empirical_outlook_label(20, 120, 75) == (
        "Strong Long-Term Edge"
    )
    assert streamlit_app.empirical_outlook_label(18, 60, 65) == (
        "Moderate Long-Term Edge"
    )
    assert streamlit_app.empirical_outlook_label(10, 60, 50) == "Weak Historical Edge"
    assert streamlit_app.empirical_outlook_label(10, 60, 58) == "Neutral"


def test_research_advisory_selects_best_risk_adjusted_horizon():
    streamlit_app = load_streamlit_app()
    empirical = streamlit_app.pd.DataFrame(
        [
            {
                "horizon": "1 week",
                "average_return_pct": 2.0,
                "win_rate_pct": 55.0,
                "observation_count": 120,
                "calibration_accuracy_pct": 70.0,
                "forecast_actual_correlation_pct": 60.0,
                "forecast_error_pct": 20.0,
            },
            {
                "horizon": "3 months",
                "average_return_pct": 18.0,
                "win_rate_pct": 70.0,
                "observation_count": 120,
                "calibration_accuracy_pct": 80.0,
                "forecast_actual_correlation_pct": 70.0,
                "forecast_error_pct": 10.0,
            },
            {
                "horizon": "5 years",
                "average_return_pct": 40.0,
                "win_rate_pct": 52.0,
                "observation_count": 120,
                "calibration_accuracy_pct": 55.0,
                "forecast_actual_correlation_pct": 50.0,
                "forecast_error_pct": 45.0,
            },
        ]
    )

    advisory = streamlit_app.advisory_row_from_empirical("AMZN", empirical)

    assert advisory["recommended_holding_period"] == "3 months"
    assert advisory["historical_win_rate"] == 70.0
    assert advisory["confidence_level"] in {"High", "Good", "Moderate"}
    assert advisory["duration_penalty"] == 1.6
    assert advisory["recommended_horizon_score"] == advisory["risk_adjusted_return"]
    assert "medium-term opportunity" in advisory["advisory_summary"]


def test_research_advisory_uses_duration_penalty_and_confidence_caps():
    streamlit_app = load_streamlit_app()

    assert streamlit_app.advisory_duration_penalty("1 week") == 1.0
    assert streamlit_app.advisory_duration_penalty("5 years") == 10.0
    assert streamlit_app.normalized_advisory_score(100.0, "High") == 100.0
    assert streamlit_app.normalized_advisory_score(100.0, "Moderate") == 85.0
    assert streamlit_app.normalized_advisory_score(100.0, "Low") == 70.0

    empirical = streamlit_app.pd.DataFrame(
        [
            {
                "horizon": "6 weeks",
                "average_return_pct": 9.0,
                "win_rate_pct": 72.0,
                "observation_count": 180,
                "calibration_accuracy_pct": 82.0,
                "forecast_actual_correlation_pct": 74.0,
                "forecast_error_pct": 9.0,
            },
            {
                "horizon": "5 years",
                "average_return_pct": 100.0,
                "win_rate_pct": 58.0,
                "observation_count": 180,
                "calibration_accuracy_pct": 52.0,
                "forecast_actual_correlation_pct": 45.0,
                "forecast_error_pct": 48.0,
            },
        ]
    )

    advisory = streamlit_app.advisory_row_from_empirical("NVDA", empirical)

    assert advisory["recommended_holding_period"] == "6 weeks"
    assert advisory["duration_penalty"] == 1.3


def test_research_advisory_scores_are_normalized_across_candidates():
    streamlit_app = load_streamlit_app()
    frame = streamlit_app.pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommended_holding_period": "6 weeks",
                "recommended_horizon_score": 5.0,
                "confidence_level": "High",
                "advisory_score": 20.0,
                "advisory_action": "Watchlist",
            },
            {
                "ticker": "MSFT",
                "recommended_holding_period": "3 months",
                "recommended_horizon_score": 10.0,
                "confidence_level": "Moderate",
                "advisory_score": 30.0,
                "advisory_action": "Watchlist",
            },
            {
                "ticker": "META",
                "recommended_holding_period": "12 months",
                "recommended_horizon_score": 8.0,
                "confidence_level": "Low",
                "advisory_score": 40.0,
                "advisory_action": "Watchlist",
            },
        ]
    )

    normalized = streamlit_app.normalize_advisory_scores(frame)

    assert normalized.loc[0, "advisory_percentile"] == 0.0
    assert normalized.loc[1, "advisory_percentile"] == 100.0
    assert normalized.loc[2, "advisory_percentile"] == 50.0
    assert normalized.loc[0, "advisory_score"] == 0.0
    assert normalized.loc[1, "advisory_score"] == 100.0
    assert normalized.loc[2, "advisory_score"] == 48.57
    assert normalized.loc[1, "conviction_level"] == "Very High Conviction"
    assert normalized.loc[1, "position_size_guidance"] == "8-12%"


def test_percentile_scores_create_advisory_score_separation():
    streamlit_app = load_streamlit_app()
    frame = streamlit_app.pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "recommended_holding_period": ["1 week"] * 5,
            "recommended_horizon_score": [10.0, 20.0, 30.0, 40.0, 50.0],
            "trading_score": [12.0, 24.0, 36.0, 48.0, 60.0],
            "swing_score": [11.0, 22.0, 33.0, 44.0, 55.0],
            "compounder_score": [9.0, 18.0, 27.0, 36.0, 45.0],
            "confidence_level": ["High"] * 5,
        }
    )

    normalized = streamlit_app.normalize_advisory_scores(frame)

    assert normalized["advisory_percentile"].tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert normalized["advisory_score"].tolist() == [0.0, 25.0, 48.57, 70.0, 100.0]
    assert normalized["trading_score"].tolist() == [0.0, 25.0, 48.57, 70.0, 100.0]
    assert normalized["trading_conviction_level"].tolist()[-1] == "Very High Conviction"
    assert normalized["trading_position_size_guidance"].tolist()[-1] == "8-12%"
    assert normalized["advisory_score"].nunique() == 5


def test_score_multiple_tickers_returns_sorted_screener_frame(monkeypatch):
    streamlit_app = load_streamlit_app()

    scores = {
        "AAPL": 72.0,
        "MSFT": 81.0,
        "META": 65.0,
    }

    def fake_safe_price_history(*args, **kwargs):
        return streamlit_app.sample_price_history(periods=30), True

    def fake_score_for_app(ticker, prices, is_fallback):
        score = SimpleNamespace(
            ticker=ticker,
            overall_score=scores[ticker],
            recommendation="Buy" if scores[ticker] >= 80 else "Watchlist",
            short_score=scores[ticker] - 1,
            medium_score=scores[ticker],
            long_score=scores[ticker] + 1,
            risk_score=20.0,
        )
        return score, {"fallback_data_used": is_fallback}

    monkeypatch.setattr(streamlit_app, "safe_price_history", fake_safe_price_history)
    monkeypatch.setattr(streamlit_app, "score_for_app", fake_score_for_app)

    frame = streamlit_app.score_multiple_tickers(
        ["AAPL", "MSFT", "META"],
        streamlit_app.pd.DataFrame(),
        min_observations=1,
    )

    assert frame["ticker"].tolist() == ["MSFT", "AAPL", "META"]
    assert {
        "ticker",
        "master_rank_score",
        "alpha_consistency_score",
        "advisory_score",
        "advisory_action",
        "advisory_percentile",
        "recommended_holding_period",
        "expected_return_range",
        "historical_win_rate",
        "confidence_level",
        "conviction_level",
        "position_size_guidance",
        "recommended_horizon_score",
        "risk_adjusted_return",
        "duration_penalty",
        "risk_reward_ratio",
        "forecast_uniqueness_score",
        "identical_forecast_flag",
        "forecast_multiplier",
        "forecast_confidence_band",
        "expected_benchmark_return",
        "expected_alpha",
        "overall_score",
        "label",
        "short_term_score",
        "medium_term_score",
        "long_term_score",
        "risk_score",
        "empirical_12m_return",
        "empirical_3y_return",
        "empirical_5y_return",
        "empirical_win_rate",
        "confidence_pct",
        "best_holding_period",
        "expected_value_12m",
        "expected_value_3y",
        "expected_value_5y",
        "best_expected_value",
        "empirical_outlook",
        "momentum_explosion_score",
        "short_term_opportunity_score",
        "opportunity_horizon",
        "empirical_1w_return",
        "empirical_2w_return",
        "empirical_4w_return",
        "empirical_6w_return",
        "empirical_3m_return",
        "empirical_6m_return",
        "trading_score",
        "trading_percentile",
        "trading_action",
        "trading_horizon",
        "trading_expected_return",
        "trading_expected_annualized_return",
        "trading_expected_benchmark_return",
        "trading_expected_alpha",
        "trading_forecast_confidence_band",
        "trading_conviction_level",
        "trading_position_size_guidance",
        "swing_score",
        "swing_percentile",
        "swing_action",
        "swing_horizon",
        "swing_expected_return",
        "swing_expected_annualized_return",
        "swing_expected_benchmark_return",
        "swing_expected_alpha",
        "swing_forecast_confidence_band",
        "swing_conviction_level",
        "swing_position_size_guidance",
        "compounder_score",
        "compounder_percentile",
        "compounder_action",
        "compounder_horizon",
        "compounder_expected_return",
        "compounder_expected_annualized_return",
        "compounder_expected_benchmark_return",
        "compounder_expected_alpha",
        "compounder_forecast_confidence_band",
        "compounder_conviction_level",
        "compounder_position_size_guidance",
    }.issubset(frame.columns)


def test_security_specific_forecasts_differentiate_same_bucket_tickers(monkeypatch):
    streamlit_app = load_streamlit_app()
    tickers = ["AMD", "NVDA", "PANW", "CRWD"]

    def fake_safe_price_history(*args, **kwargs):
        return streamlit_app.sample_price_history(periods=260), True

    def fake_score_for_app(ticker, prices, is_fallback):
        return SimpleNamespace(
            ticker=ticker,
            overall_score=82.0,
            recommendation="Buy",
            short_score=78.0,
            medium_score=82.0,
            long_score=86.0,
            risk_score=24.0,
            components={},
        ), {}

    def fake_empirical_score_bucket_forecast(score, validation_records, min_observations=1):
        return streamlit_app.pd.DataFrame(
            [
                {
                    "horizon": "1 week",
                    "average_return_pct": 2.0,
                    "win_rate_pct": 58.0,
                    "observation_count": 400,
                    "calibration_accuracy_pct": 82.0,
                    "forecast_actual_correlation_pct": 76.0,
                    "forecast_error_pct": 8.0,
                },
                {
                    "horizon": "3 months",
                    "average_return_pct": 12.0,
                    "win_rate_pct": 66.0,
                    "observation_count": 400,
                    "calibration_accuracy_pct": 82.0,
                    "forecast_actual_correlation_pct": 76.0,
                    "forecast_error_pct": 8.0,
                },
                {
                    "horizon": "12 months",
                    "average_return_pct": 22.0,
                    "win_rate_pct": 72.0,
                    "observation_count": 400,
                    "calibration_accuracy_pct": 82.0,
                    "forecast_actual_correlation_pct": 76.0,
                    "forecast_error_pct": 8.0,
                },
                {
                    "horizon": "3 years",
                    "average_return_pct": 80.0,
                    "win_rate_pct": 78.0,
                    "observation_count": 400,
                    "calibration_accuracy_pct": 82.0,
                    "forecast_actual_correlation_pct": 76.0,
                    "forecast_error_pct": 8.0,
                },
                {
                    "horizon": "5 years",
                    "average_return_pct": 140.0,
                    "win_rate_pct": 84.0,
                    "observation_count": 400,
                    "calibration_accuracy_pct": 82.0,
                    "forecast_actual_correlation_pct": 76.0,
                    "forecast_error_pct": 8.0,
                },
            ]
        )

    monkeypatch.setattr(streamlit_app, "safe_price_history", fake_safe_price_history)
    monkeypatch.setattr(streamlit_app, "score_for_app", fake_score_for_app)
    monkeypatch.setattr(
        streamlit_app,
        "empirical_score_bucket_forecast",
        fake_empirical_score_bucket_forecast,
    )

    frame = streamlit_app.score_multiple_tickers(
        tickers,
        streamlit_app.pd.DataFrame(),
        sort_by="forecast_uniqueness_score",
    )

    assert frame["expected_return_range"].nunique() == len(tickers)
    assert frame["forecast_uniqueness_score"].min() == 100.0
    assert streamlit_app.forecast_uniqueness_ratio(frame) > 90.0
    assert not frame["identical_forecast_flag"].any()
    assert frame["forecast_confidence_band"].isin(
        ["Very High", "High", "Moderate", "Low"]
    ).all()
    assert (frame["expected_alpha"] != 0).all()


def test_identical_security_forecasts_are_flagged():
    streamlit_app = load_streamlit_app()
    frame = streamlit_app.pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "expected_return_range": ["10.0% to 12.0%", "10.0% to 12.0%", "7.0% to 9.0%"],
            "trading_expected_return_range": ["2.0% to 3.0%"] * 3,
            "swing_expected_return_range": ["5.0% to 6.0%"] * 3,
            "compounder_expected_return_range": ["20.0% to 25.0%"] * 3,
        }
    )

    checked = streamlit_app.apply_forecast_uniqueness_scores(frame)

    assert bool(checked.loc[0, "identical_forecast_flag"]) is True
    assert bool(checked.loc[1, "identical_forecast_flag"]) is True
    assert bool(checked.loc[2, "identical_forecast_flag"]) is False
    assert checked.loc[0, "forecast_uniqueness_score"] < 100.0


def test_score_multiple_tickers_can_sort_by_best_expected_value(monkeypatch):
    streamlit_app = load_streamlit_app()

    scores = {"AAPL": 60.0, "MSFT": 90.0}

    def fake_safe_price_history(*args, **kwargs):
        return streamlit_app.sample_price_history(periods=30), True

    def fake_score_for_app(ticker, prices, is_fallback):
        return SimpleNamespace(
            ticker=ticker,
            overall_score=scores[ticker],
            recommendation="Watchlist",
            short_score=scores[ticker],
            medium_score=scores[ticker],
            long_score=scores[ticker],
            risk_score=20.0,
        ), {}

    def fake_empirical_score_bucket_forecast(score, validation_records, min_observations=1):
        five_year_return = 350.0 if score == 60.0 else 10.0
        return streamlit_app.pd.DataFrame(
            [
                {
                    "horizon": "12 months",
                    "average_return_pct": 5.0,
                    "win_rate_pct": 60.0,
                    "observation_count": 100,
                    "calibration_accuracy_pct": 70.0,
                    "forecast_actual_correlation_pct": 60.0,
                    "forecast_error_pct": 20.0,
                },
                {
                    "horizon": "3 years",
                    "average_return_pct": 30.0,
                    "win_rate_pct": 65.0,
                    "observation_count": 100,
                    "calibration_accuracy_pct": 70.0,
                    "forecast_actual_correlation_pct": 60.0,
                    "forecast_error_pct": 20.0,
                },
                {
                    "horizon": "5 years",
                    "average_return_pct": five_year_return,
                    "win_rate_pct": 80.0,
                    "observation_count": 100,
                    "calibration_accuracy_pct": 70.0,
                    "forecast_actual_correlation_pct": 60.0,
                    "forecast_error_pct": 20.0,
                },
            ]
        )

    monkeypatch.setattr(streamlit_app, "safe_price_history", fake_safe_price_history)
    monkeypatch.setattr(streamlit_app, "score_for_app", fake_score_for_app)
    monkeypatch.setattr(
        streamlit_app,
        "empirical_score_bucket_forecast",
        fake_empirical_score_bucket_forecast,
    )

    frame = streamlit_app.score_multiple_tickers(
        ["AAPL", "MSFT"],
        streamlit_app.pd.DataFrame(),
        sort_by="best_expected_value",
    )

    assert frame["ticker"].tolist() == ["AAPL", "MSFT"]


def test_opportunity_finder_filters_and_sorts_results():
    streamlit_app = load_streamlit_app()
    screener_frame = streamlit_app.pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "master_rank_score": 72.0,
                "short_term_opportunity_score": 70.0,
                "confidence_pct": 65.0,
            },
            {
                "ticker": "MSFT",
                "master_rank_score": 88.0,
                "short_term_opportunity_score": 90.0,
                "confidence_pct": 75.0,
            },
            {
                "ticker": "META",
                "master_rank_score": 96.0,
                "short_term_opportunity_score": 95.0,
                "confidence_pct": 30.0,
            },
        ]
    )

    result = streamlit_app.opportunity_finder_results(
        screener_frame,
        minimum_score=70,
        minimum_confidence=60,
        limit=20,
    )

    assert result["ticker"].tolist() == ["MSFT", "AAPL"]


def test_portfolio_builder_weights_and_summary():
    streamlit_app = load_streamlit_app()
    screener_frame = streamlit_app.pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "overall_score": 80.0,
                "empirical_outlook": "Strong Long-Term Edge",
                "expected_value_5y": 22000.0,
                "empirical_12m_return": 20.0,
                "empirical_3y_return": 70.0,
                "empirical_5y_return": 120.0,
            },
            {
                "ticker": "MSFT",
                "overall_score": 40.0,
                "empirical_outlook": "Neutral",
                "expected_value_5y": 15000.0,
                "empirical_12m_return": 10.0,
                "empirical_3y_return": 30.0,
                "empirical_5y_return": 50.0,
            },
        ]
    )

    screener_frame["short_term_opportunity_score"] = [40.0, 90.0]
    screener_frame["opportunity_horizon"] = ["Long-Term Investment", "Swing Trade"]
    screener_frame["confidence_pct"] = [80.0, 60.0]

    portfolio = streamlit_app.portfolio_builder_frame(
        screener_frame,
        mode="Long-Term Portfolio",
    )
    short_portfolio = streamlit_app.portfolio_builder_frame(
        screener_frame,
        mode="Short-Term Opportunity Portfolio",
    )
    balanced_portfolio = streamlit_app.portfolio_builder_frame(
        screener_frame,
        mode="Balanced Portfolio",
    )
    summary = streamlit_app.portfolio_builder_summary(
        screener_frame,
        mode="Long-Term Portfolio",
    )

    assert portfolio["weight"].round(4).tolist() == [0.6667, 0.3333]
    assert short_portfolio["weight"].round(4).tolist() == [0.3077, 0.6923]
    assert balanced_portfolio["weight"].round(4).tolist() == [0.48, 0.52]
    assert summary["strongest_holding"] == "AAPL"
    assert summary["weakest_holding"] == "MSFT"
    assert summary["portfolio_horizon_classification"] == "Long-Term Investment"
    assert round(summary["expected_portfolio_12m_return"], 2) == 16.67


def test_best_ideas_portfolio_allocation_and_summary():
    streamlit_app = load_streamlit_app()
    screener_frame = streamlit_app.pd.DataFrame(
        [
            {
                "ticker": "TRADE1",
                "trading_score": 100.0,
                "trading_conviction_level": "Very High Conviction",
                "trading_expected_return": 4.0,
                "trading_expected_return_range": "3.0% to 5.0%",
                "trading_win_rate": 62.0,
                "trading_horizon": "2 weeks",
                "swing_score": 20.0,
                "swing_conviction_level": "Avoid",
                "swing_expected_return": 1.0,
                "swing_expected_return_range": "0.0% to 2.0%",
                "swing_win_rate": 50.0,
                "swing_horizon": "3 months",
                "compounder_score": 30.0,
                "compounder_conviction_level": "Speculative",
                "compounder_expected_return": 20.0,
                "compounder_expected_return_range": "17.0% to 23.0%",
                "compounder_win_rate": 55.0,
                "compounder_horizon": "12 months",
                "confidence_pct": 80.0,
                "historical_downside": 2.0,
            },
            {
                "ticker": "SWING1",
                "trading_score": 30.0,
                "trading_conviction_level": "Speculative",
                "trading_expected_return": 1.0,
                "trading_expected_return_range": "0.0% to 2.0%",
                "trading_win_rate": 52.0,
                "trading_horizon": "1 week",
                "swing_score": 100.0,
                "swing_conviction_level": "High Conviction",
                "swing_expected_return": 12.0,
                "swing_expected_return_range": "10.0% to 14.0%",
                "swing_win_rate": 68.0,
                "swing_horizon": "6 months",
                "compounder_score": 20.0,
                "compounder_conviction_level": "Avoid",
                "compounder_expected_return": 15.0,
                "compounder_expected_return_range": "13.0% to 17.0%",
                "compounder_win_rate": 51.0,
                "compounder_horizon": "12 months",
                "confidence_pct": 70.0,
                "historical_downside": 6.0,
            },
            {
                "ticker": "COMP1",
                "trading_score": 20.0,
                "trading_conviction_level": "Avoid",
                "trading_expected_return": 0.5,
                "trading_expected_return_range": "-1.5% to 2.5%",
                "trading_win_rate": 49.0,
                "trading_horizon": "1 week",
                "swing_score": 30.0,
                "swing_conviction_level": "Speculative",
                "swing_expected_return": 5.0,
                "swing_expected_return_range": "3.0% to 7.0%",
                "swing_win_rate": 55.0,
                "swing_horizon": "3 months",
                "compounder_score": 100.0,
                "compounder_conviction_level": "Moderate Conviction",
                "compounder_expected_return": 45.0,
                "compounder_expected_return_range": "38.2% to 51.8%",
                "compounder_win_rate": 75.0,
                "compounder_horizon": "5 years",
                "confidence_pct": 65.0,
                "historical_downside": 15.0,
            },
        ]
    )

    aggressive = streamlit_app.best_ideas_portfolio_frame(
        screener_frame,
        "Aggressive Growth",
        per_sleeve=1,
    )
    balanced = streamlit_app.best_ideas_portfolio_frame(
        screener_frame,
        "Balanced Growth",
        per_sleeve=1,
    )
    conservative = streamlit_app.best_ideas_portfolio_frame(
        screener_frame,
        "Conservative Compounder",
        per_sleeve=1,
    )
    summary = streamlit_app.best_ideas_portfolio_summary(aggressive)
    buy_today = streamlit_app.best_ideas_for_today(screener_frame)

    assert streamlit_app.conviction_position_size_pct("Very High Conviction") == 10.0
    assert streamlit_app.conviction_position_size_pct("High Conviction") == 7.0
    assert streamlit_app.conviction_position_size_pct("Moderate Conviction") == 5.0
    assert round(float(aggressive["weight"].sum()), 2) == 100.0
    assert aggressive.groupby("sleeve")["weight"].sum().round(2).to_dict() == {
        "Compounder": 20.0,
        "Swing": 30.0,
        "Trading": 50.0,
    }
    assert balanced.groupby("sleeve")["weight"].sum().round(2).to_dict() == {
        "Compounder": 40.0,
        "Swing": 40.0,
        "Trading": 20.0,
    }
    assert conservative.groupby("sleeve")["weight"].sum().round(2).to_dict() == {
        "Compounder": 70.0,
        "Swing": 20.0,
        "Trading": 10.0,
    }
    assert round(summary["expected_portfolio_return"], 2) == 14.6
    assert round(summary["expected_portfolio_win_rate"], 2) == 66.4
    assert round(summary["expected_drawdown"], 2) == 5.8
    assert round(summary["risk_reward_ratio"], 2) == 2.52
    assert buy_today["Ticker"].tolist() == ["TRADE1", "SWING1", "COMP1"]


def test_prediction_accuracy_dashboard_core_path_runs_with_fallback_data():
    streamlit_app = load_streamlit_app()

    prices = {
        ticker: streamlit_app.sample_price_history(start="2013-01-01", end="2026-01-01")
        for ticker in ["AAPL", "MSFT"]
    }
    flags = {ticker: True for ticker in prices}
    frame = streamlit_app.point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker=flags,
        step_days=252,
    )
    accuracy = streamlit_app.prediction_accuracy_metrics(frame)
    threshold = streamlit_app.score_threshold_validation(frame)
    calibration = streamlit_app.pit_forecast_calibration(frame)

    assert not frame.empty
    assert "average_prediction_error_pct" in accuracy
    assert not threshold.empty
    assert not calibration.empty
    assert frame["sample_fundamentals_used"].eq(False).all()


def test_streamlit_forecast_tab_empirical_core_path_runs_with_fallback_data():
    streamlit_app = load_streamlit_app()

    prices = streamlit_app.sample_price_history(periods=300)
    score, quality = streamlit_app.score_for_app("META", prices, True)
    report, *_ = streamlit_app.forecast_frames(score, prices)
    validation_records = streamlit_app.point_in_time_prediction_frame(
        {
            ticker: streamlit_app.sample_price_history(
                start="2013-01-01",
                end="2026-01-01",
            )
            for ticker in ["AAPL", "MSFT", "META"]
        },
        start_date="2016-01-01",
        fallback_data_used_by_ticker={"AAPL": True, "MSFT": True, "META": True},
        step_days=252,
    )
    empirical = streamlit_app.empirical_score_bucket_forecast(
        score.overall_score,
        validation_records,
        investment_amount=10,
        min_observations=1,
    )
    outlook = streamlit_app.empirical_outlook_interpretation(empirical)
    outcome = streamlit_app.empirical_investment_outcome_table(empirical)

    assert quality["fallback_data_used"] is True
    assert report.ticker == "META"
    assert not validation_records.empty
    assert not empirical.empty
    assert "expected_value" in empirical.columns
    assert outlook["headline"].startswith("Empirical Outlook:")
    assert not outcome.empty
