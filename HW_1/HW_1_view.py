import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from HW_1_config import config

def plot_data(data, title="", column="Close"):
    """
    Построение линейного графика ценовых данных.
    :param data: Датафрейм с данными.
    :param title: Заголовок графика.
    :param column: Колонка для отображения.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data.index, data[column], label=column+" price")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

# Список тикеров акций из индекса S&P 500 для просмотра
sp500_tickers = ['AAPL']

# Список криптовалют для просмотра
cryptos = ['BTC-USD']

# Загрузка всех файлов из директории
files = os.listdir(config.Directory)
full_paths = [os.path.join(config.Directory, f) for f in files]
files = [f for f in full_paths if os.path.isfile(f)]

# Фильтруем список файлов, которых необходимо просмотреть
files = [
    file for file in files
    if any(ticker in file for ticker in sp500_tickers + cryptos)
]


dataFrames = {}
for fileName in files:
    df = pd.read_csv(fileName)
    df = df.dropna()
    dataFrames[fileName] = df
    # Вывод графика свечей
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
    fig.update_layout(
        title=dict(text=fileName)
    )
    fig.show()
    # Вывод линейного графика
    df.index = pd.to_datetime(df['Date'], utc=True)
    plot_data(df, fileName, "Open")
# Вывод всех линейных графиков сразу
plt.show()
