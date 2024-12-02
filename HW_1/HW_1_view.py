import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import talib
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
dataMACD = {}
for fileName in files:
    df = pd.read_csv(fileName)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.dropna()
    dataFrames[fileName] = df
    macd_df = df.copy() #копирование для построения MACD
    macd_df['MACD'], macd_df['MACD_Signal'], macd_df['MACD_Hist'] = talib.MACD(macd_df["Close"], fastperiod=12,
                                                                               slowperiod=26, signalperiod=9)
    dataMACD[fileName] = macd_df
    # Вывод графика свечей инструмента
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'))
    # Добавление MACD в отдельной области
    fig.add_trace(go.Scatter(x=macd_df['Date'], y=macd_df['MACD'], mode='lines', name='MACD', line=dict(color='blue'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=macd_df['Date'], y=macd_df['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red'), yaxis='y2'))
    fig.add_trace(go.Bar(x=macd_df['Date'], y=macd_df['MACD_Hist'], name='MACD Histogram', marker=dict(color='darkviolet', opacity=0.6), yaxis='y2'))

    # Обновление параметров графика
    fig.update_layout(
        title=dict(text=fileName),
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(type='date'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='MACD', overlaying='y', side='right', showgrid=False, position=1.0),
        showlegend=True
    )

    # Вывод графика
    fig.show()

    # Вывод линейного графика инструмента
    df.index = pd.to_datetime(df['Date'], utc=True)
    plot_data(df, fileName, "Open")

# Вывод всех линейных графиков сразу
plt.show()
