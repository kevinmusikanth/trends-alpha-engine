from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import sleep

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class CompanyProfile:
    ticker: str
    name: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None


class YahooFinanceError(RuntimeError):
    """Raised when Yahoo Finance data cannot be fetched after retries."""


class YahooFinanceConnector:
    """Yahoo Finance connector for free price and company metadata."""

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 1.0) -> None:
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _retry(self, operation):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return operation()
            except Exception as error:  # yfinance raises several transport-specific errors.
                last_error = error
                if attempt < self.max_retries - 1:
                    sleep(self.backoff_seconds * (2**attempt))
        raise YahooFinanceError(str(last_error)) from last_error

    def fetch_price_history(
        self,
        ticker: str,
        start: str | date | None = None,
        end: str | date | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        history = self._retry(
            lambda: yf.Ticker(ticker).history(
                start=start,
                end=end,
                period=period,
                auto_adjust=False,
            )
        )
        if history.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        history = history.reset_index()
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        history = history.rename(columns=rename_map)
        available = [column for column in rename_map.values() if column in history.columns]
        return history[available].copy()

    def fetch_profile(self, ticker: str) -> CompanyProfile:
        info = self._retry(lambda: yf.Ticker(ticker).get_info())
        return CompanyProfile(
            ticker=ticker.upper(),
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
        )

    def fetch_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        stock = yf.Ticker(ticker)
        return {
            "income_statement": stock.financials,
            "quarterly_income_statement": stock.quarterly_financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
        }
