import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def backtest_dca(prices_df: pd.DataFrame, weights: np.ndarray, weekly_investment: float = 100.0) -> Dict[str, Any]:
    weights = np.where(weights < 0, 0, weights)
    weights = weights / np.sum(weights)

    investment_per_asset = weekly_investment * weights

    shares_bought = prices_df.rdiv(investment_per_asset, axis=1).fillna(0)

    accumulated_shares = shares_bought.cumsum()

    portfolio_value_per_asset = accumulated_shares * prices_df.fillna(0)
    portfolio_value = portfolio_value_per_asset.sum(axis=1)

    actual_investment_per_week = (shares_bought > 0).astype(int) * investment_per_asset
    total_invested = actual_investment_per_week.sum(axis=1).cumsum()

    # ROI
    final_balance = portfolio_value.iloc[-1]
    total_invested_final = total_invested.iloc[-1]
    total_roi = (final_balance - total_invested_final) / total_invested_final if total_invested_final > 0 else 0

    # Max Drawdown
    # Считаем просадку от пика стоимости портфеля
    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min()

    # Sortino Ratio (weekly data)
    returns = portfolio_value.pct_change().dropna()
    downside_returns = returns[returns < 0]
    if len(returns) > 0 and len(downside_returns) > 0 and downside_returns.std() != 0:
        sortino_ratio = returns.mean() / downside_returns.std()
    else:
        sortino_ratio = 0

    return {
        'portfolio_value': portfolio_value,
        'final_balance': final_balance,
        'total_invested': total_invested_final,
        'total_roi': total_roi,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'accumulated_shares': accumulated_shares.iloc[-1], # Итоговое количество монет
        'prices_last': prices_df.iloc[-1].fillna(0) # Последние цены для информативности
    }
