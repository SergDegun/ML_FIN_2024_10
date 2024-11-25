import pandas as pd

# Директория для файлов с данными
directory = 'Data'

def LoadTickersFromCsv(tickers):
    dataFrames = {}
    for ticker in tickers:
        fileName = f"{directory}\\{ticker}.csv"
        df = pd.read_csv(fileName)
        #df = df.dropna()
        dataFrames[ticker] = df
    return dataFrames

# Список тикеров акций из индекса S&P 500
sp500_tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']
stock_data = LoadTickersFromCsv(sp500_tickers)

# Список криптовалют
cryptos_tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
crypto_data = LoadTickersFromCsv(cryptos_tickers)


# Этап 2: Проверка наличия пропусков и ошибок

def check_outliers(data, column="Close"):
    """
    Определение выбросов в данных с использованием межквартильного размаха.

    :param data: Датафрейм с данными.
    :param column: Название колонки, по которой проводится анализ.
    :return: Датафрейм с метками выбросов.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers


def analyze_outliers(data, outliers, threshold=0.05):
    """
    Анализ выбросов и определение их природы.

    :param data: Датафрейм с данными.
    :param outliers: Множество индексов выбросов.
    :param threshold: Порог для определения значимости выброса.
    :return: Датафрейм с пометкой реальных данных и выбросов.
    """
    analyzed_data = data.copy()
    analyzed_data["Outlier"] = False
    analyzed_data.loc[outliers, "Outlier"] = True
    real_data = analyzed_data.query("Outlier == False")
    outlier_data = analyzed_data.query("Outlier == True")

    IsRealValues = len(outlier_data) / len(analyzed_data) <= threshold
    '''if IsRealValues == True:
        print("Выбросы скорее всего являются реальными данными.")
    else:
        print("Выбросы могут быть ошибочными данными.")
    '''
    return IsRealValues


def check_dataFrames(dataFrames):
    # Проверка пропусков
    for ticker, ticker_data in dataFrames.items():
        missing_data = ticker_data.isna().sum().sum()

        if missing_data > 0:
            print(f"{ticker}: Обнаружены пропуски в данных.")
        else:
            print(f"{ticker}: Пропусков в данных нет.")

        # Проверка выбросов
        outliers_open = check_outliers(ticker_data, "Open")
        if (outliers_open.sum() > 0):
            if (analyze_outliers(ticker_data, outliers_open) == False):
                print(f"{ticker}: обнаружены выбросы в Open")
        outliers_high = check_outliers(ticker_data, "High")
        if (outliers_high.sum() > 0):
            if (analyze_outliers(ticker_data, outliers_high) == False):
                print(f"{ticker}: обнаружены выбросы в High")
        outliers_low = check_outliers(ticker_data, "Low")
        if (outliers_low.sum() > 0):
            if (analyze_outliers(ticker_data, outliers_low) == False):
                print(f"{ticker}: обнаружены выбросы в Low")
        outliers_close = check_outliers(ticker_data, "Close")
        if (outliers_close.sum() > 0):
            if (analyze_outliers(ticker_data, outliers_close) == False):
                print(f"{ticker}: обнаружены выбросы в Close")
        outliers_volume = check_outliers(ticker_data, "Volume")
        if (outliers_volume.sum() > 0):
            if (analyze_outliers(ticker_data, outliers_volume) == False):
                print(f"{ticker}: обнаружены выбросы в Volume")


print(f"Проверка акций из индекса S&P 500:")
check_dataFrames(stock_data)

print(f"Проверка криптовалют:")
check_dataFrames(crypto_data)
