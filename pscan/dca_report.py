import pandas as pd
import numpy as np
import os
from pscan.engine import backtest_dca

def main():
    # Настройка путей
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    csv_file = os.path.join(DATA_DIR, "crypto_data.csv")

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    # Загрузка данных
    try:
        prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Очистка данных
    prices_df = prices_df.dropna(how='all')
    
    # Фильтрация только активных на текущий момент монет (как в main.py)
    last_prices = prices_df.iloc[-1].fillna(0)
    active_symbols = last_prices[last_prices > 0].index.tolist()
    
    results = []
    
    print(f"Calculating DCA ROI for {len(active_symbols)} coins...")
    
    for symbol in active_symbols:
        # Получаем данные только по одной монете
        coin_prices = prices_df[[symbol]].dropna()
        
        if len(coin_prices) < 2:
            continue
            
        # Для одной монеты вес 1.0
        # Используем упрощенную логику DCA или вызываем backtest_dca
        # backtest_dca ожидает DataFrame с ценами и массив весов
        
        # Инвестируем 100 USDT каждую неделю (или каждый шаг в CSV)
        weekly_investment = 100.0
        shares_bought = (weekly_investment / coin_prices[symbol]).fillna(0)
        total_shares = shares_bought.sum()
        total_invested = (shares_bought > 0).sum() * weekly_investment
        
        if total_invested == 0:
            continue
            
        final_value = total_shares * coin_prices[symbol].iloc[-1]
        roi = (final_value - total_invested) / total_invested
        
        results.append({
            'symbol': symbol,
            'roi': roi
        })
        
    # Сортировка по доходности (от большего к меньшему)
    results.sort(key=lambda x: x['roi'], reverse=True)
    
    # Вывод результатов
    print("\n" + "="*40)
    print(f"{'Coin':<15} | {'ROI (%)':<10}")
    print("-" * 40)
    for res in results:
        print(f"{res['symbol']:<15} | {res['roi']:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()
