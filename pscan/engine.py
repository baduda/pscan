import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def backtest_dca(prices_df: pd.DataFrame, weights: np.ndarray, weekly_investment: float = 100.0) -> Dict[str, Any]:
    """
    Векторизованный бэктест стратегии DCA.
    Вход: 
      - prices_df: DataFrame с ценами (строки - даты, колонки - тикеры)
      - weights: Массив весов активов (должен быть нормализован)
      - weekly_investment: Еженедельная сумма инвестиций в USDT
    Выход: 
      - Словарь с метриками и динамикой портфеля
    """
    # Гарантируем нормализацию весов (на всякий случай, хотя должно приходить из оптимизатора)
    weights = weights / np.sum(weights)
    
    # Распределяем еженедельную сумму по активам
    investment_per_asset = weekly_investment * weights
    
    # Количество купленных монет на каждой неделе
    # shares_bought = investment / price. Если цена NaN (монеты еще нет), то покупаем 0
    shares_bought = prices_df.rdiv(investment_per_asset, axis=1).fillna(0)
    
    # Накопленное количество монет
    accumulated_shares = shares_bought.cumsum()
    
    # Стоимость портфеля на каждую неделю (сумма накопленных монет * текущая цена)
    portfolio_value_per_asset = accumulated_shares * prices_df.fillna(0)
    portfolio_value = portfolio_value_per_asset.sum(axis=1)
    
    # Общая сумма инвестиций на каждую неделю
    # Считаем сколько недель существовал каждый актив для точного подсчета затрат
    active_weeks = prices_df.notna().cumsum()
    # Но проще: на каждой неделе мы инвестируем weekly_investment в портфель
    # Если на какой-то неделе монеты еще нет, ее доля (investment_per_asset) остается в кэше или не тратится?
    # ТЗ говорит: "каждую неделю симулируется покупка на фиксированную сумму USDT".
    # Если монеты нет, мы просто не тратим ее долю или перераспределяем?
    # Обычно в DCA если монеты нет, мы ее не покупаем. 
    # Посчитаем фактические затраты:
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
    
    # Sharpe Ratio (упрощенно для недельных данных)
    returns = portfolio_value.pct_change().dropna()
    if len(returns) > 0 and returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(52)  # 52 недели в году
    else:
        sharpe_ratio = 0
        
    return {
        'portfolio_value': portfolio_value,
        'final_balance': final_balance,
        'total_invested': total_invested_final,
        'total_roi': total_roi,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'accumulated_shares': accumulated_shares.iloc[-1], # Итоговое количество монет
        'prices_last': prices_df.iloc[-1].fillna(0) # Последние цены для информативности
    }
