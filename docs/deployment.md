# Deployment Guide

Version 1 is intended to run locally. Deployment comes later after data quality, backtesting correctness, and model calibration are validated.

## Local Requirements

- Python 3.11+
- PostgreSQL 14+
- Internet access for free data connectors

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Create a PostgreSQL database:

```bash
createdb tae
```

Initialize tables:

```bash
python -m tae.db.init_db
```

Launch the app:

```bash
streamlit run streamlit_app.py
```

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `TAE_DATABASE_URL` | SQLAlchemy database URL |
| `TAE_SEC_USER_AGENT` | SEC-compliant user agent string |
| `TAE_DEFAULT_UNIVERSE` | Default universe key, for example `sp500` |

## Later Deployment Options

- Single-user server with PostgreSQL and Streamlit
- Internal research dashboard behind SSO
- Scheduled data collection jobs
- Object storage for raw source snapshots
- Separate research and production databases

## Production Hardening Checklist

- Add point-in-time data sources.
- Add retry and rate-limit policies for connectors.
- Add data lineage metadata.
- Add authentication.
- Add audit logs for model changes.
- Add reproducible backtest run manifests.
- Add containerized deployment.

## Streamlit Community Cloud Deployment

This repository is prepared for Streamlit Community Cloud with:

- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`
- `streamlit_app.py` as the main app file

### Push To GitHub

Run these commands from your local Terminal:

```bash
cd /Users/kevinbodyexcel.co.za/Documents/Codex/2026-06-20/project-name-trends-alpha-engine-tae
git init -b main
git add .
git commit -m "Initial Trends Alpha Engine scaffold"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME/YOUR_REPOSITORY` with your actual GitHub repository.

### Deploy On Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Choose `New app`.
3. Select the GitHub repository.
4. Set the branch to `main`.
5. Set the main file path to `streamlit_app.py`.
6. Deploy.

No secrets are required for Version 1.
