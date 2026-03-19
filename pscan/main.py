import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
from pscan.data_loader import load_data, get_survivors
from pscan.optimizer import GeneticOptimizer
from pscan.engine import backtest_dca

# Настройка путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")

# Создание директорий, если их нет
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Настройка логирования
log_file = os.path.join(LOGS_DIR, "optimization.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 1. Параметры
    start_date = '2021-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    csv_file = os.path.join(DATA_DIR, '../data/crypto_data.csv')
    
    # Список монет для оптимизации (если None - берем все доступные)
    target_symbols = [
        # "SOL/USDT", "TRX/USDT", "BCH/USDT",
        # "XRP/USDT", "BTC/USDT", "BNB/USDT", "SUN/USDT",
        # "JST/USDT", "INJ/USDT", "FET/USDT", "HBAR/USDT", "DCR/USDT",
        # "AAVE/USDT", "DOGE/USDT", "XLM/USDT", "ETH/USDT", "DUSK/USDT",
        # "LINK/USDT", "DASH/USDT", "TRB/USDT", "OG/USDT", "ATM/USDT",
        # "ADA/USDT", "CHZ/USDT", "LTC/USDT", "STX/USDT"
    ]
    # target_symbols = None  # Раскомментируйте, чтобы использовать все монеты

    excluded_symbols = ["LUNA/USDT", "USDT/USDT", "PAXG/USDT", "EUR/USDT"] # Сбрасываем исключения, так как задан точный список


    # 2. Загрузка данных
    logger.info("Checking historical data...")
    
    # Проверка наличия кэша с USDT/USDT
    if os.path.exists(csv_file):
        try:
            # Читаем только заголовки для проверки колонок
            header_df = pd.read_csv(csv_file, index_col=0, nrows=0)
            if 'USDT/USDT' in header_df.columns:
                logger.info(f"Found valid cache with USDT/USDT. Using symbols from {csv_file}")
                # Читаем весь файл
                prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                symbols = list(prices_df.columns)
            else:
                logger.info("Cache found but 'USDT/USDT' missing. Updating data...")
                symbols = get_survivors(f"{start_date} 00:00:00")
                prices_df = load_data(symbols, start_date, end_date, csv_file)
        except Exception as e:
            logger.error(f"Error reading cache: {e}. Re-downloading...")
            symbols = get_survivors(f"{start_date} 00:00:00")
            prices_df = load_data(symbols, start_date, end_date, csv_file)
    else:
        # Динамическое получение списка символов для проверки (полная загрузка)
        symbols = get_survivors(f"{start_date} 00:00:00")
        logger.info(f"Found {len(symbols)} candidate symbols existing since {start_date}")
        
        if not symbols:
            logger.error("No symbols found! Check your connection or date.")
            return

        # load_data теперь инкрементально догружает то, чего нет
        prices_df = load_data(symbols, start_date, end_date, csv_file)
    
    # Очистка данных: убедимся, что нет пустых строк в начале/конце
    prices_df = prices_df.dropna(how='all')
    
    # ФИЛЬТРАЦИЯ ДЕЛИСТИНГОВАННЫХ МОНЕТ
    # Если последняя цена NaN или 0, значит монета больше не торгуется
    last_prices = prices_df.iloc[-1].fillna(0)
    active_symbols = last_prices[last_prices > 0].index.tolist()
    
    if len(active_symbols) < len(prices_df.columns):
        logger.info(f"Removed {len(prices_df.columns) - len(active_symbols)} delisted/dead assets.")
        prices_df = prices_df[active_symbols]

    # Список символов берется из итогового DataFrame
    symbols = list(prices_df.columns)

    # Фильтрация по списку target_symbols, если он указан
    if target_symbols:
        # Оставляем только те символы, которые есть и в списке, и в данных
        available_targets = [s for s in target_symbols if s in symbols]
        if not available_targets:
            logger.warning("None of the target symbols were found in the data. Using all available assets.")
        else:
            prices_df = prices_df[available_targets]
            symbols = list(prices_df.columns)
            logger.info(f"Filtered to {len(symbols)} target symbols.")

    # Исключение монет из excluded_symbols
    if excluded_symbols:
        existing_excluded = [s for s in excluded_symbols if s in symbols]
        if existing_excluded:
            prices_df = prices_df.drop(columns=existing_excluded)
            symbols = list(prices_df.columns)
            logger.info(f"Excluded {len(existing_excluded)} symbols: {existing_excluded}")

    # Проверка количества доступных активов после всех фильтров
    if len(symbols) < 2:
        if len(symbols) == 0:
            logger.error("No symbols left after filtering/exclusions. Aborting optimization.")
        else:
            logger.error("Only 1 symbol left after filtering/exclusions. GA requires at least 2 assets. Aborting optimization.")
        return

    logger.info(f"Using {len(symbols)} symbols for optimization")
    
    logger.info(f"Data period: {prices_df.index[0]} to {prices_df.index[-1]}")
    logger.info(f"Number of assets: {len(prices_df.columns)}")
    
    # 3. Оптимизация
    logger.info("Starting Genetic Algorithm Optimization...")
    ga_instance = GeneticOptimizer(prices_df).run()
    
    # 4. Анализ результатов
    solution, fitness, idx = ga_instance.best_solution()
    solution = np.where(solution < 0, 0, solution)
    best_weights = solution / np.sum(solution)
    
    logger.info("\n" + "="*30)
    logger.info("BEST PORTFOLIO WEIGHTS")
    logger.info("="*30)
    for i, symbol in enumerate(prices_df.columns):
        if best_weights[i] > 0.001:  # Выводим веса > 0.1%
            logger.info(f"{symbol}: {best_weights[i]:.2%}")
            
    # 5. Бэктест лучшего решения
    final_result = backtest_dca(prices_df, best_weights)
    
    logger.info("\n" + "="*30)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*30)
    logger.info(f"Final Balance: {final_result['final_balance']:.2f} USDT")
    logger.info(f"Total Invested: {final_result['total_invested']:.2f} USDT")
    logger.info(f"Total ROI: {final_result['total_roi']:.2%}")
    logger.info(f"Max Drawdown: {final_result['max_drawdown']:.2%}")
    logger.info(f"Sortino Ratio: {final_result['sortino_ratio']:.4f}")

    logger.info("\n" + "="*30)
    logger.info("FINAL PORTFOLIO COMPOSITION")
    logger.info("="*30)
    logger.info(f"{'Asset':<12} | {'Weight':<8} | {'Amount':<12} | {'Value (USDT)':<12}")
    logger.info("-" * 55)
    
    acc_shares = final_result['accumulated_shares']
    last_prices = final_result['prices_last']
    
    for i, symbol in enumerate(prices_df.columns):
        if best_weights[i] > 0.001:
            amount = acc_shares[symbol]
            val_usdt = amount * last_prices[symbol]
            logger.info(f"{symbol:<12} | {best_weights[i]:<8.2%} | {amount:<12.6f} | {val_usdt:<12.2f}")
    
    # 6. Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(final_result['portfolio_value'].index, final_result['portfolio_value'].values, label='Portfolio Value (Best Weights)', color='blue')
    
    # Рисуем линию накопленных инвестиций
    shares_bought = prices_df.rdiv(100.0 * best_weights, axis=1).fillna(0)
    actual_investment_per_week = (shares_bought > 0).astype(int) * (100.0 * best_weights)
    total_invested_series = actual_investment_per_week.sum(axis=1).cumsum()
    
    plt.plot(total_invested_series.index, total_invested_series.values, label='Total Invested', color='red', linestyle='--')
    
    plt.title('DCA Portfolio Optimization Results')
    plt.xlabel('Date')
    plt.ylabel('USDT')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'optimization_result.png'))
    logger.info(f"\nPlot saved as {os.path.join(PLOTS_DIR, 'optimization_result.png')}")
    logger.info(f"Detailed logs saved to {log_file}")
    plt.show()

if __name__ == "__main__":
    main()
