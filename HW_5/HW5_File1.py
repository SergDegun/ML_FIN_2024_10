import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import streamlit as st


# Загрузка данных из файла
def load_data_from_file(file_path):
  try:
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data
  except FileNotFoundError:
      print(f"Файл '{file_path}' не найден.")
      return None

# Загрузка данных из yaho finance
def load_data_from_yf(ticker, timeframe='1d'):
  df = yf.download(ticker, interval=timeframe)
  df = df.reset_index()
  if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.droplevel(level=1)
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  return df

# Индикатор RSI
def rsi(close_prices, n=14):
  delta = close_prices.diff().dropna()
  up = delta.clip(lower=0)
  down = -delta.clip(upper=0)
  rs = up.ewm(span=n, adjust=False).mean() / down.ewm(span=n, adjust=False).mean()
  return 100 - (100 / (1 + rs))

# Колонки для нормализации
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Mean', 'Value']
# Колонки, которые не нужно нормализовать
columns_not_to_scale = ['SMA', 'RSI']

# Предобработка данных
def preprocess_data(data, window_size=30):
  # Добавление новых признаков
  data['Mean'] = (data['Open']+data['High']+data['Low']+data['Close']) / 4 # Средняя цена
  data['Value'] = data['Volume'] * data['Mean'] # Объём в деньгах
  data['SMA'] = data['Close'].rolling(window_size).mean()  # Скользящая средняя
  data['RSI'] = rsi(data['Close'], n=window_size)  # Индекс относительной силы

  # Удаляем строки с NaN-значениями
  data.dropna(inplace=True)

  # Нормализация данных только к выбранным колонкам
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(data[columns_to_scale])

  # Преобразуем результат обратно в DataFrame
  scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

  # Добавляем колонки, которые не нужно было нормализовать
  scaled_df[columns_not_to_scale] = data[columns_not_to_scale].reset_index(drop=True)

  # Преобразуем DataFrame в массив NumPy для удобства работы
  scaled_data = scaled_df.values

  # Создание временных окон
  X, y = [], []
  for i in range(window_size, len(scaled_data)):
      X.append(scaled_data[i-window_size:i, :])
      y.append(scaled_data[i, 0]) #Целевая переменная
  X, y = np.array(X), np.array(y)

  # Разделение на обучающую и тестовую выборки
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  return X_train, X_test, y_train, y_test, scaler

def create_lstm_model(input_shape):
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
  model.add(Dropout(0.2))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dropout(0.2))
  model.add(Dense(25))
  model.add(Dense(1))  # Прогнозирование цены
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))  # Прогнозирование цены
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
  return history

def test_model(model, X_test, y_test):
    # Получаем прогнозы
    predictions = model.predict(X_test)
    return predictions, y_test

def calculate_metrics(predictions, y_test):
  # метрика: точность направления движения цены
  pred_sign = np.sign(predictions[1:] - predictions[:-1])
  test_sign = np.sign(y_test[1:] - y_test[:-1])
  direction_accuracy = accuracy_score(test_sign, pred_sign)
  rmse = np.mean((predictions - y_test) ** 2) ** 0.5
  mae = np.mean(np.abs(predictions - y_test))
  average = y_test.mean()
  rmse2avg = rmse / average
  mae2avg = mae / average
  return direction_accuracy, rmse, mae, rmse2avg, mae2avg

# Загрузка данных
#file_path = 'stock_data.csv'  # Замените на путь к вашему файлу
#data = load_data_from_file(file_path)
data = load_data_from_yf("AAPL")

# Предобработка данных
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Создание и обучение модели
model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
history = train_model(model, X_train, y_train, 10)

# Тестирование модели
predictions, y_test = test_model(model, X_test, y_test, scaler)

# Расчет метрик
direction_accuracy, rmse, mae, rmse2avg, mae2avg  = calculate_metrics(predictions, y_test)
print(f"Точность направления движения цены: {direction_accuracy:.4f}")
print(f"Средняя абсолютная ошибка цены (MAE): {mae:.4f}")
print(f"Средняя квадратичная ошибка цены (RMSE): {rmse:.4f}")
print(f"Относительная средняя абсолютная ошибка цены (MAE): {mae2avg:.4f}")
print(f"Относительная средняя квадратичная ошибка цены (RMSE): {rmse2avg:.4f}")

# Сохранение метрик в файл
metrics = {
    'Модель': 'LSTM',
    'Точность направления': direction_accuracy,
    'Средняя абсолютная ошибка': mae,
    'Средняя квадратичная ошибка': rmse,
    'Относительная средняя абсолютная ошибка': mae2avg,
    'Относительная средняя квадратичная ошибка': rmse2avg
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('metrics.csv', index=False)

# Создание дашборда с использованием Streamlit
def create_dashboard(data, predictions, y_test):
    st.title('Дашборд торговой стратегии')
    st.write("### График прогнозов и реальных цен")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Реальные цены')
    plt.plot(predictions, label='Прогнозы')
    plt.legend()
    st.pyplot(plt)

    st.write("### Метрики модели")
    metrics_df = pd.read_csv('metrics.csv')
    st.table(metrics_df)

create_dashboard(data, predictions, y_test)

def simulate_trading(model, X_test, y_test, initial_capital=10000, commission=0.001):
    """
    Симуляция торговли на тестовых данных.
    Будем покупать акции, если модель предсказывает рост цены, и продавать, если предсказывает падение
    :param model: Обученная модель
    :param X_test: Тестовые данные (временные окна)
    :param y_test: Реальные цены
    :param initial_capital: Начальный капитал
    :param commission: Комиссия за сделку (например, 0.1%)
    :return: История капитала, список сделок
    """
    # Получаем прогнозы модели
    predictions = model.predict(X_test)

    # Инициализация переменных
    capital = initial_capital
    position = 0  # Текущая позиция (0 - нет позиции, 1 - куплено, -1 - продано)
    trades = []  # Список для хранения сделок
    capital_history = []  # История изменения капитала

    for i in range(len(predictions) - 1):
        current_price = y_test[i]
        next_price = y_test[i + 1]
        predicted_change = predictions[i + 1] - predictions[i]

        # Сигнал на покупку (предсказание роста)
        if predicted_change > 0 and position <= 0:
            if position == -1:
                # Закрываем короткую позицию (покупаем)
                capital += position * current_price * (1 - commission)
                position = 0
            # Покупаем
            shares_to_buy = capital // (current_price * (1 + commission))
            if shares_to_buy > 0:
                capital -= shares_to_buy * current_price * (1 + commission)
                position = 1
                trades.append(('buy', current_price, shares_to_buy))

        # Сигнал на продажу (предсказание падения)
        elif predicted_change < 0 and position >= 0:
            if position == 1:
                # Закрываем длинную позицию (продаем)
                capital += position * current_price * (1 - commission)
                position = 0
            # Продаем (шорт)
            shares_to_sell = capital // (current_price * (1 + commission))
            if shares_to_sell > 0:
                capital += shares_to_sell * current_price * (1 - commission)
                position = -1
                trades.append(('sell', current_price, shares_to_sell))

        # Обновляем историю капитала
        if position == 1:
            capital_history.append(capital + (shares_to_buy * next_price))
        elif position == -1:
            capital_history.append(capital - (shares_to_sell * next_price))
        else:
            capital_history.append(capital)

    # Закрываем последнюю позицию, если она открыта
    if position == 1:
        capital += position * y_test[-1] * (1 - commission)
    elif position == -1:
        capital += position * y_test[-1] * (1 - commission)

    return capital_history, trades

# Симуляция торгов
capital_history, trades = simulate_trading(model, X_test, y_test, initial_capital=10000, commission=0.001)
# Вывод результатов
print(f"Конечный капитал: {capital_history[-1]:.2f}")
print(f"Количество сделок: {len(trades)}")
print(f"Первые 5 сделок: {trades[:5]}")

# Добавление графика капитала в дашборд
def create_dashboard2(data, predictions, y_test, capital_history):
    st.title('Дашборд торговой стратегии')
    st.write("### График прогнозов и реальных цен")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test, label='Реальные цены')
    ax.plot(predictions, label='Прогнозы')
    ax.legend()
    st.pyplot(fig)

    st.write("### График изменения капитала")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(capital_history, label='Капитал')
    ax.set_xlabel('Время')
    ax.set_ylabel('Капитал')
    ax.legend()
    st.pyplot(fig)

    st.write("### Метрики модели")
    metrics_df = pd.read_csv('metrics.csv')
    st.table(metrics_df)

create_dashboard2(data, predictions, y_test, capital_history)
