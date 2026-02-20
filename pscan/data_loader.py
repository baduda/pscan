import ccxt
import pandas as pd
import time
from datetime import datetime
from typing import List, Optional
import os

def get_survivors(start_date: str = "2021-01-01 00:00:00") -> List[str]:
    """
    Ищет спотовые пары к USDT на Binance, которые существуют с указанной даты.
    """
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    
    # Ищем только спотовые пары к USDT
    symbols = [s for s in exchange.symbols if '/USDT' in s and markets[s]['spot']]
    
    # Исключаем UP/DOWN токены
    exclude_keywords = ['UP/', 'DOWN/', 'BEAR/', 'BULL/']
    symbols = [s for s in symbols if not any(kw in s for kw in exclude_keywords)]
    
    start_ts = exchange.parse8601(start_date)
    
    survivors = []
    
    print(f"Checking {len(symbols)} symbols...")
    for symbol in symbols:
        try:
            # Запрашиваем самую первую свечу для этого символа
            first_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1w', since=0, limit=1)
            if first_ohlcv and first_ohlcv[0][0] <= start_ts:
                survivors.append(symbol)
                print(f"[OK] {symbol} exists since 2021")
        except Exception:
            continue
            
    return survivors

def fetch_ohlcv(symbol: str, timeframe: str = '1w', since: Optional[int] = None) -> pd.DataFrame:
    """
    Скачивает историю OHLCV для указанного тикера.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # инкремент времени для следующего запроса
            
            # Если получили меньше лимита, значит данные закончились
            if len(ohlcv) < limit:
                break
                
            # Небольшая пауза для соблюдения rate limit
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def load_data(symbols: List[str], start_date: str, end_date: str, filename: str = 'crypto_data.csv') -> pd.DataFrame:
    """
    Скачивает цены Close для списка тикеров и сохраняет в CSV инкрементально.
    """
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_dt = pd.to_datetime(end_date)
    
    # Пытаемся загрузить существующий файл
    if os.path.exists(filename):
        try:
            combined_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Loaded existing data with {len(combined_df.columns)} assets.")
            # Гарантируем наличие стабильной колонки 'USDT/USDT' со значением 1.0
            if not combined_df.empty and 'USDT/USDT' not in combined_df.columns:
                combined_df['USDT/USDT'] = 1.0
                combined_df.to_csv(filename)
                print("Added synthetic stablecoin column 'USDT/USDT' to existing CSV.")
        except Exception as e:
            print(f"Error loading existing CSV: {e}. Starting fresh.")
            combined_df = pd.DataFrame()
    else:
        combined_df = pd.DataFrame()

    for symbol in symbols:
        if symbol in combined_df.columns:
            print(f"Data for {symbol} already exists. Skipping...")
            continue
            
        print(f"Fetching data for {symbol}...")
        df = fetch_ohlcv(symbol, '1w', since)
        if not df.empty:
            close_prices = df['close'].rename(symbol)
            # Ограничиваем по конечной дате
            close_prices = close_prices[close_prices.index <= end_dt]
            
            if combined_df.empty:
                combined_df = pd.DataFrame(close_prices)
            else:
                # Объединяем по индексу (датам)
                combined_df = combined_df.join(close_prices, how='outer')
            
            # Сохраняем после каждой успешной загрузки для предотвращения потери данных
            combined_df.to_csv(filename)
            print(f"Saved {symbol} to CSV.")
        else:
            print(f"No data returned for {symbol}.")
            
    # В конце гарантируем наличие стабильной колонки 'USDT/USDT' со значением 1.0
    if not combined_df.empty and 'USDT/USDT' not in combined_df.columns:
        combined_df['USDT/USDT'] = 1.0
        combined_df.to_csv(filename)
        print("Ensured synthetic stablecoin column 'USDT/USDT' exists in CSV.")
            
    return combined_df

if __name__ == "__main__":
    # Динамическое получение списка выживших монет
    survivors_list = get_survivors()
    print(f"Total survivor coins: {len(survivors_list)}")
    
    if survivors_list:
        load_data(survivors_list, '2021-01-01', datetime.now().strftime('%Y-%m-%d'))
