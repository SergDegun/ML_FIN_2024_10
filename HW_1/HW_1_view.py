import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

def plot_data(data, title="", column="Close"):
    """
    Построение графика ценовых данных.
    :param data: Датафрейм с данными.
    :param title: Заголовок графика.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data.index, data[column], label=column+" Price")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend()
    plt.show()

# Директория для файлов с данными
directory = 'Data'

# Список тикеров акций из индекса S&P 500 для просмотра
sp500_tickers = ['AAPL']

# Список криптовалют для просмотра
cryptos = ['BTC-USD']

# Загрузка всех файлов из директории
files = os.listdir(directory)
full_paths = [os.path.join(directory, f) for f in files]
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
    #df['Date'] = pd.to_datetime(str(df['Date']), format='%YYYY-%MM-%dd')
    #df['Date'] = str(df['Date'])[:10]
    dataFrames[fileName] = df
    fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
    fig.update_layout(
        title=dict(text=fileName)
    )
    fig.show()
    # Нарисовать линейным графиком цену закрытия
    plot_data(df, fileName, "Open")
