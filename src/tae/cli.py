from __future__ import annotations

import argparse
from dataclasses import asdict

from tae.backtesting.engine import banded_forward_returns, forward_returns
from tae.connectors.yahoo import YahooFinanceConnector
from tae.scoring.engine import score_ticker


def score_command(args: argparse.Namespace) -> None:
    connector = YahooFinanceConnector()
    for ticker in args.tickers:
        prices = connector.fetch_price_history(ticker, period="1y")
        score = score_ticker(ticker, prices)
        print(asdict(score))


def backtest_command(args: argparse.Namespace) -> None:
    connector = YahooFinanceConnector()
    prices = connector.fetch_price_history(args.ticker, start=args.start, end=args.end)
    returns = forward_returns(prices)
    returns["score"] = 50.0
    summary = banded_forward_returns(returns, horizon=args.horizon)
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trends Alpha Engine local CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score one or more tickers")
    score_parser.add_argument("tickers", nargs="+")
    score_parser.set_defaults(func=score_command)

    backtest_parser = subparsers.add_parser("backtest", help="Run a simple forward-return backtest")
    backtest_parser.add_argument("ticker")
    backtest_parser.add_argument("--start", required=True)
    backtest_parser.add_argument("--end", required=True)
    backtest_parser.add_argument("--horizon", default="1m")
    backtest_parser.set_defaults(func=backtest_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

