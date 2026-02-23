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
    log_file = os.path.join(LOGS_DIR, "dca_start_finder.log")
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
    last_prices = prices_df.iloc[-1].fillna(0)
    active_symbols = last_prices[last_prices > 0].index.tolist()
    
    results = []
    
    logger.info(f"Анализ оптимальной точки входа для {len(active_symbols)} монет...")
    logger.info("Условие: Цена < Средняя цена DCA (если бы начали покупать с самого начала)")
    logger.info("-" * 80)
    
    for symbol in active_symbols:
        # Получаем данные по одной монете, удаляем NaN в начале
        coin_series = prices_df[symbol].dropna()
        
        if len(coin_series) < 2:
            continue
            
        # Симулируем накопление DCA с самого начала истории данных монеты
        # На каждом шаге считаем среднюю цену (Total Spent / Total Shares)
        
        investment = 100.0 # Фиксированная сумма на покупку
        
        shares_bought = investment / coin_series
        cum_shares = shares_bought.cumsum()
        cum_spent = (shares_bought > 0).astype(int).cumsum() * investment
        
        avg_dca_price = cum_spent / cum_shares
        
        # Находим первый момент, когда текущая цена падает НИЖЕ средней цены DCA
        # Это и будет сигналом к началу "реальных" покупок
        entry_signals = coin_series < avg_dca_price
        
        if entry_signals.any():
            first_entry_date = entry_signals.idxmax()
            entry_price = coin_series.loc[first_entry_date]
            dca_price_at_entry = avg_dca_price.loc[first_entry_date]
            
            results.append({
                'symbol': symbol,
                'date': first_entry_date,
                'price': entry_price,
                'dca_avg': dca_price_at_entry,
                'drop_pct': (entry_price / dca_price_at_entry - 1)
            })
        else:
            # Если цена никогда не падала ниже DCA (например, постоянный рост)
            pass

    # Сортировка по проценту (наибольшее отклонение в начале)
    results.sort(key=lambda x: x['drop_pct'])
    
    # Вывод результатов
    logger.info(f"{'Монета':<15} | {'Дата входа':<12} | {'Цена':<10} | {'DCA Средняя':<12} | {'Отклонение':<10}")
    logger.info("-" * 80)
    for res in results:
        date_str = res['date'].strftime('%Y-%m-%d')
        logger.info(f"{res['symbol']:<15} | {date_str:<12} | {res['price']:<10.4f} | {res['dca_avg']:<12.4f} | {res['drop_pct']:>9.2%}")
    
    logger.info("-" * 80)
    logger.info(f"Всего монет с точкой входа: {len(results)} из {len(active_symbols)}")
    logger.info(f"Отчет сохранен в logs/dca_start_finder.log")

if __name__ == "__main__":
    main()
