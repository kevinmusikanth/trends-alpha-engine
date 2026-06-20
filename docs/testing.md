# Testing Documentation

## Test Strategy

TAE has three important testing layers:

1. Unit tests for scoring math, risk scoring, and return calculations.
2. Integration tests for connectors and database persistence.
3. Research validation tests for backtesting correctness and look-ahead-bias prevention.

## Run Tests

```bash
pip install -e ".[dev]"
pytest
```

## Required Test Coverage Areas

### Scoring

- Component weights sum to 100 for Model A, Model B, and Model C.
- Missing data produces conservative scores instead of crashes.
- Inputs are clipped to valid score ranges.
- Overall score combines short, medium, long, narrative, capital flow, surprise, and risk consistently.

### Backtesting

- Forward return windows are calculated from future prices only.
- Score-band labels are correct.
- Win rate uses positive forward return.
- Volatility and Sharpe handle empty or zero-variance returns.
- Maximum drawdown handles both rising and falling curves.

### Data Connectors

- Yahoo Finance connector normalizes column names.
- SEC connector requires a compliant user agent.
- Connector failures are returned as structured errors where possible.

### Bias Controls

Backtests must avoid:

- Look-ahead bias
- Survivorship bias
- Rebalanced universe leakage
- Using revised financial data as if it were known on the historical score date

Version 1 includes code structure for these controls. A production-grade research run should use point-in-time datasets when available.

## Manual QA

Before a release:

1. Run a small ticker score in the CLI.
2. Open the Streamlit app locally.
3. Search for a ticker.
4. Run a short historical backtest.
5. Confirm all outputs include the research-only disclaimer.

