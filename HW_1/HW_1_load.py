import yfinance as yf
from pathlib import Path
from HW_1_config import config

# Этап 1: Загрузка котировок

def LoadSaveTickers(tickers, directory):
    tickers_data = {}
    for ticker in tickers:
        tickers_data[ticker] = yf.Ticker(ticker).history(period="max")
        filename = f"{directory}\\{ticker}.csv"
        tickers_data[ticker].to_csv(filename)
        print(f"Сохранено: {filename}")
    return tickers_data

# Создание директории, если её не существует
file_path = Path(f"{config.Directory}\\ticker.csv")
file_path.parent.mkdir(parents=True, exist_ok=True)

# Список тикеров акций из индекса S&P 500 для загрузки
sp500_tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']

# Загрузка данных по акциям
print(f"Загрузка и сохранение акций из индекса S&P 500:")
stock_data = LoadSaveTickers(sp500_tickers, config.Directory)

# Список тикеров криптовалют для загрузки
cryptos_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']

# Загрузка данных по криптовалютам
print(f"Загрузка и сохранение криптовалют:")
crypto_data = LoadSaveTickers(cryptos_tickers, config.Directory)
