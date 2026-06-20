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


def get_universe(name: str) -> list[str]:
    key = name.lower().replace(" ", "")
    if key in {"sp500", "s&p500"}:
        return SP500_SAMPLE
    if key in {"nasdaq100", "nasdaq"}:
        return NASDAQ100_SAMPLE
    if key in {"russell2000", "russell"}:
        return RUSSELL2000_SAMPLE
    raise ValueError(f"Unknown universe: {name}")

