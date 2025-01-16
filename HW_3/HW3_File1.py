import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import time
import os


def clean_data(df, zbals = 8):
    # Удаляем выбросы по каждому числовому столбцу
    for col in df.select_dtypes(include='number', exclude='datetime64'):
        z_scores = zscore(df[col])
        outliers = abs(z_scores) > zbals
        df.loc[outliers, col] = None

    # Заполняем пропущенные значения медианными значениями только для числовых столбцов
    numeric_cols = df.select_dtypes(include='number').columns
    medians = df[numeric_cols].median()  # Расчитываем медианы

    # Используем fillna только для числовых столбцов
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    return df


def create_new_features(df, IsNeedStandarting = False):
    # Создаем новый признак, например, отношение двух столбцов
    df['Open2Close'] = df['Open'] / df['Close']
    df['Average'] = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4
    df['Return'] = df['Average'].pct_change()  # Возврат
    df['Moving_average'] = df['Average'].rolling(window=5).mean()  # Скользящее среднее
    df['Volatility'] = df['Average'].rolling(window=5).std()  # Волатильность

    # Стандартизация данных
    if IsNeedStandarting:
        scaler = StandardScaler()
        # Обработка только числовых колонок
        numeric_cols = df.select_dtypes(include='number').columns
        df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    return df


def data_pipeline(filename, new_data):
    # Очистка данных от выбросов и заполнение пропусков
    new_data = clean_data(new_data)

    # Формирование новых признаков
    new_data = create_new_features(new_data)

    # Чтение имеющихся данных из файла или базы данных
    IsUpdate = False
    # Проверка существования файла AAPL_data.csv
    if os.path.exists(filename):
        # Загрузка существующего файла
        old_data = pd.read_csv(filename, parse_dates=['Date'])

        old_data['Date'] = pd.to_datetime(old_data['Date'])

        # Находим даты, которые уже существуют в старом файле
        existing_dates = set(old_data['Date'].dt.date)

        # Оставляем только те строки из new_df, которых нет в old_df
        missing_rows = new_data[~new_data['Date'].dt.date.isin(existing_dates)]

        if len(missing_rows) > 0:
            # Добавляем недостающие строки к старому DataFrame
            combined_df = pd.concat([old_data, missing_rows], ignore_index=True)

            # Сортируем по дате после добавления новых строк
            combined_df.sort_values(by='Date', inplace=True)

            # Сохранение обработанных данных обратно в файл или базу данных
            combined_df.to_csv(filename, index=False)
            # Флаг наличия обновления
            IsUpdate = True
    else:
        # Если итогового файла не существует, создаем его с новыми данными
        new_data.to_csv(filename, index=False)
        # Флаг наличия обновления
        IsUpdate = True

    return IsUpdate


# Функция для запуска дашборда по одному из признаков
def run_dashboard(filename, feature_name = 'Open2Close'):
    # Загрузка данных
    data = pd.read_csv(filename)
    # Преобразование даты в индекс
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    #data.index = pd.to_datetime(data.index)

    # Проверка на наличие колонки feature_name
    if feature_name in data.columns:
        # Построение графика распределения
        plt.figure(figsize=(10, 6))

        # Описание значений
        values = data[feature_name].values
        plt.hist(values, bins=30, alpha=0.6, color='blue', density=True)

        # Построение KDE
        kde = stats.gaussian_kde(values)
        x = np.linspace(min(values), max(values), 100)
        plt.plot(x, kde(x), color='red')

        plt.title('Распределение ' + feature_name)
        plt.xlabel(feature_name)
        plt.ylabel('Плотность')
        plt.grid(True)

        plt.show()
        #plt.savefig(f'D:\\my_plot.png')  # Сохранит график в формате PNG

    st.title("Анализ накопленных данных")

    # График распределения значений одного из признаков
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x=feature_name, kde=True, ax=ax)
    st.pyplot(fig)

    # Таблица с данными
    st.write(data.head())

    # Графики корреляции между признаками
    # Расчет корреляционной матрицы только для числовых столбцов
    numeric_cols = data.select_dtypes(include='number').columns
    corr_matrix = data[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# Пример работы с новыми данными
def main(ticker):
    new_data = yf.download(ticker, interval='1d')
    new_data = new_data.reset_index()
    new_data.columns = new_data.columns.droplevel(level=1)
    #new_data = pd.read_csv(f'Data\\AAPL.csv', parse_dates=['Date'])

    #Проверка наличия столбцов
    #new_data.columns = new_data.columns.str.lower() #Названия столбцов в нижний регистр
    if ('Date' not in new_data.columns or 'Open' not in new_data.columns or 'High' not in new_data.columns
            or 'Low' not in new_data.columns or 'Close' not in new_data.columns or 'Volume' not in new_data.columns):
        return

    #new_data.index = pd.to_datetime(new_data.index)
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data = new_data.dropna()  #Удаление пустых записей

    filename = 'Data\\'+ticker+'_data.csv'
    IsUpdate = data_pipeline(filename, new_data)
    if IsUpdate == True:
        run_dashboard(filename)

if __name__ == "__main__":
    #main("AAPL")
    while True:
        main("AAPL")
        time.sleep(60)  # Проверка новых данных каждую минуту
