import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

def main():
    # Настройка путей
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")
    csv_file = os.path.join(DATA_DIR, "crypto_data.csv")
    
    # Создание директории логов, если ее нет
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Настройка логирования
    log_file = os.path.join(LOGS_DIR, "dca_ranker.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', # Упрощенный формат для таблиц
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(csv_file):
        logger.error(f"Ошибка: Файл {csv_file} не найден.")
        return

    # Загрузка данных
    try:
        prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    except Exception as e:
        logger.error(f"Ошибка при чтении CSV: {e}")
        return

    # Очистка данных
    prices_df = prices_df.dropna(how='all')
    
    # Фильтрация только активных на текущий момент монет
    # Монета считается активной, если её последняя цена не NaN и больше 0
    last_prices = prices_df.iloc[-1]
    active_symbols = last_prices[last_prices > 0].index.tolist()
    
    results = []
    
    logger.info(f"Анализ DCA для {len(active_symbols)} монет...")
    logger.info("Откл. от DCA = (Текущая цена / Средняя цена DCA) - 1")
    logger.info("Сред. откл. вниз = Среднее значение отрицательных отклонений цены от текущей на тот момент средней DCA")
    logger.info("Умный DCA = Профит стратегии: покупка только если Цена < DCA Средняя * (1 + Сред. откл. вниз)")
    logger.info("-" * 135)
    
    for symbol in active_symbols:
        # Получаем данные по одной монете, удаляем NaN в начале (до листинга)
        coin_series = prices_df[symbol].dropna()
        
        if len(coin_series) < 1:
            continue
            
        # Симулируем накопление DCA с самого начала истории данных монеты
        # На каждом шаге считаем среднюю цену (Total Spent / Total Shares)
        
        investment = 100.0 # Фиксированная сумма на покупку
        
        # 1. Обычный DCA для расчета метрик
        shares_bought = investment / coin_series
        cum_shares = shares_bought.cumsum()
        cum_spent = (shares_bought > 0).astype(int).cumsum() * investment
        avg_dca_price = cum_spent / cum_shares
        
        current_price = coin_series.iloc[-1]
        current_dca_avg = avg_dca_price.iloc[-1]
        
        # Отклонение от DCA
        dca_deviation = (current_price / current_dca_avg - 1)
        
        # Среднее отклонение вниз
        all_deviations = (coin_series / avg_dca_price - 1)
        downward_deviations = all_deviations[all_deviations < 0]
        avg_downward_deviation = downward_deviations.mean() if not downward_deviations.empty else 0
        
        # 2. Умный DCA: покупаем только если цена < DCA * (1 + avg_downward_deviation)
        # Для честности используем avg_downward_deviation, рассчитанный на предыдущем шаге, 
        # но для простоты ранжирования можно использовать финальный показатель как фильтр стратегии.
        # Пользователь просил "если она стоит меньше чем дца минус среднее отклонение".
        
        threshold_mult = (1 + avg_downward_deviation)
        buy_mask = coin_series < (avg_dca_price * threshold_mult)
        
        smart_shares = (investment / coin_series[buy_mask]).cumsum()
        smart_spent = buy_mask.astype(int).cumsum() * investment
        
        if not smart_shares.empty and smart_spent.iloc[-1] > 0:
            last_smart_shares = smart_shares.iloc[-1]
            last_smart_spent = smart_spent.iloc[-1]
            smart_profit = (last_smart_shares * current_price / last_smart_spent) - 1
        else:
            smart_profit = 0
            
        results.append({
            'symbol': symbol,
            'current_price': current_price,
            'dca_avg': current_dca_avg,
            'dca_deviation': dca_deviation,
            'avg_downward_dev': avg_downward_deviation,
            'smart_profit': smart_profit
        })

    results.sort(key=lambda x: -x['smart_profit'])
    
    # Вывод результатов
    logger.info(f"{'Монета':<15} | {'Тек. Цена':<10} | {'DCA Средняя':<12} | {'Откл. от DCA':<12} | {'Сред. откл. вниз':<16} | {'Умный DCA':<12}")
    logger.info("-" * 135)
    for res in results:
        logger.info(f"{res['symbol']:<15} | {res['current_price']:<10.4f} | {res['dca_avg']:<12.4f} | {res['dca_deviation']:>12.2%} | {res['avg_downward_dev']:>16.2%} | {res['smart_profit']:>12.2%}")
    
    logger.info("-" * 135)
    logger.info(f"Всего проанализировано монет: {len(results)}")
    logger.info(f"Отчет сохранен в logs/dca_ranker.log")

if __name__ == "__main__":
    main()
