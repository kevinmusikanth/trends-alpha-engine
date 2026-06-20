from tae.connectors.fallback import sample_price_history
from tae.scoring.engine import score_ticker as real_score_ticker
from tae.forecasting.point_in_time import (
    forecast_calibration,
    investment_outcome_validation,
    point_in_time_prediction_frame,
    prediction_accuracy_metrics,
    score_threshold_validation,
)


def point_in_time_prices():
    return {
        ticker: sample_price_history(start="2013-01-01", end="2026-01-01")
        for ticker in ["AAPL", "MSFT", "META"]
    }


def test_point_in_time_prediction_frame_uses_only_historical_slices():
    prices = point_in_time_prices()
    frame = point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )

    assert not frame.empty
    assert {
        "date",
        "ticker",
        "score",
        "forecast_return",
        "actual_future_return",
        "prediction_error",
    }.issubset(frame.columns)
    assert frame["sample_fundamentals_used"].eq(False).all()


def test_point_in_time_prediction_frame_handles_older_score_signature(monkeypatch):
    from tae.forecasting import point_in_time

    def older_score_ticker(
        ticker,
        price_history,
        manual_features=None,
        live_price_data_available=None,
        fallback_data_used=False,
    ):
        return real_score_ticker(
            ticker,
            price_history,
            manual_features=manual_features,
            live_price_data_available=live_price_data_available,
            fallback_data_used=fallback_data_used,
        )

    monkeypatch.setattr(point_in_time, "score_ticker", older_score_ticker)
    prices = {"META": sample_price_history(start="2013-01-01", end="2026-01-01")}

    frame = point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={"META": True},
        step_days=252,
    )

    assert not frame.empty
    assert frame["sample_fundamentals_used"].eq(False).all()


def test_prediction_accuracy_and_threshold_validation_are_calculated():
    prices = point_in_time_prices()
    frame = point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )
    metrics = prediction_accuracy_metrics(frame)
    threshold = score_threshold_validation(frame)
    calibration = forecast_calibration(frame)

    assert "average_prediction_error_pct" in metrics
    assert "forecast_actual_correlation" in metrics
    assert not threshold.empty
    assert {"threshold", "horizon", "average_actual_return_pct"}.issubset(
        threshold.columns
    )
    assert not calibration.empty
    assert {"forecast_bucket", "average_actual_return_pct"}.issubset(calibration.columns)


def test_point_in_time_investment_outcome_validation_compares_benchmarks():
    prices = point_in_time_prices()
    frame = point_in_time_prediction_frame(
        prices,
        start_date="2016-01-01",
        fallback_data_used_by_ticker={ticker: True for ticker in prices},
        step_days=252,
    )
    benchmark = frame.rename(columns={"actual_future_return": "actual_return"})

    outcome = investment_outcome_validation(
        frame,
        investment_amount=10,
        benchmarks={"S&P 500": benchmark},
    )

    assert not outcome.empty
    assert {
        "portfolio_value",
        "cagr_pct",
        "sharpe_ratio",
        "maximum_drawdown_pct",
        "S&P 500_portfolio_value",
    }.issubset(outcome.columns)
