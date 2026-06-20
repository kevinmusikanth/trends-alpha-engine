# Trends Alpha Engine (TAE)

Trends Alpha Engine is a local-first research platform for stock scoring, ranking, screening, watchlists, and historical backtesting.

It is not a trading bot, not financial advice, and has no brokerage integration. Version 1 uses free data sources only.

## Goals

TAE is designed to find statistical edges across three research horizons:

| Model | Horizon | Target Return |
| --- | --- | --- |
| Short-Term Alpha | 1 to 4 weeks | 12%+ |
| Medium-Term Alpha | 1 to 4 months | 15%+ |
| Long-Term Compounder | 4 to 12 months | 30%+ |

## Version 1 Scope

- Yahoo Finance price and company metadata connector
- SEC company facts connector
- Scoring engine for the three alpha models
- Narrative, capital flow, and surprise factor hooks
- Historical forward-return backtesting
- Score-band analytics
- Screener and watchlist primitives
- Hypothetical portfolio testing
- Streamlit local app
- PostgreSQL storage layer
- Documentation for architecture, testing, and deployment

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Start PostgreSQL locally and create a database named `tae`, or update `TAE_DATABASE_URL` in `.env`.

```bash
python -m tae.db.init_db
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

Use these deployment settings:

| Setting | Value |
| --- | --- |
| Repository | Your GitHub repository for this project |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version | `3.11` from `runtime.txt` |
| Dependencies | Installed from `requirements.txt` |

No paid API keys are required for Version 1. If you later add private database credentials or API keys, store them in Streamlit Cloud secrets rather than committing them to GitHub.

## Example CLI Usage

```bash
python -m tae.cli score AAPL MSFT NVDA
python -m tae.cli backtest AAPL --start 2021-01-01 --end 2024-12-31
```

## Project Layout

```text
src/tae/
  connectors/      Free data connectors
  db/              SQLAlchemy models and setup
  scoring/         Model A, B, C and composite scoring
  backtesting/     Forward-return and score-band analytics
  screening/       Ticker search and recommendation logic
  watchlists/      Watchlist and score-change tracking
  portfolio/       Hypothetical portfolio analytics
  cli.py           Local command entrypoint
streamlit_app.py   Local UI
docs/              Architecture, testing, deployment docs
tests/             Unit tests
```

## Important Disclaimer

TAE produces research signals and historical statistics. Scores are probabilistic research features, not investment recommendations. Any output marked `Buy`, `Watch`, or `Avoid` is a platform label for screening priority, not financial advice.

## Documentation

- [Architecture](docs/architecture.md)
- [Testing](docs/testing.md)
- [Deployment](docs/deployment.md)
