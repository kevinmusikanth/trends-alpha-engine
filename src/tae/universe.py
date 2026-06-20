SP500_SAMPLE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
]

NASDAQ100_SAMPLE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "COST",
    "ADBE",
    "NFLX",
]

RUSSELL2000_SAMPLE = [
    "SMCI",
    "CELH",
    "ELF",
    "MSTR",
    "TMDX",
]

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def get_universe(name: str) -> list[str]:
    key = name.lower().replace(" ", "")
    if key in {"sp500", "s&p500"}:
        return public_equity_universe(SP500_WIKIPEDIA_URL, "Symbol", SP500_SAMPLE)
    if key in {"nasdaq100", "nasdaq"}:
        return public_equity_universe(NASDAQ100_WIKIPEDIA_URL, "Ticker", NASDAQ100_SAMPLE)
    if key in {"russell2000", "russell"}:
        return RUSSELL2000_SAMPLE
    raise ValueError(f"Unknown universe: {name}")


def public_equity_universe(url: str, ticker_column: str, fallback: list[str]) -> list[str]:
    try:
        import pandas as pd

        tables = pd.read_html(url)
        for table in tables:
            if ticker_column in table.columns:
                tickers = table[ticker_column].dropna().astype(str).str.upper().tolist()
                return [ticker.replace(".", "-") for ticker in tickers]
    except Exception:
        pass
    return fallback
