from __future__ import annotations

import numpy as np
import pandas as pd


def win_rate(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, annualization: float = 252) -> float:
    returns = returns.dropna()
    std = returns.std()
    if returns.empty or std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(annualization))


def return_summary(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "average_return": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
    return {
        "average_return": float(returns.mean()),
        "win_rate": win_rate(returns),
        "volatility": float(returns.std()),
        "max_drawdown": max_drawdown(returns),
        "sharpe_ratio": sharpe_ratio(returns),
    }

