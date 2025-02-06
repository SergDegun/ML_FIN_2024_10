import pandas as pd
import numpy as np
import yfinance as yf
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score as evs
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import talib

# Простая реализация процесса торговли для тестирования стратегии
def TradingSimple(strategy_data, commission = 0.0, is_short_enable = False):
    result_money = 1.0
    result_instr = 0.0
    trades_count = 0
    commission_money = 0.0
    # Проход по всем позициям
    for index, row in strategy_data.iterrows():
        if row['Signal'] == 1 and  result_instr <= 0:
            if (result_instr < 0):
                # Выход в CASHE
                money_val = result_instr * row['Close']
                result_money += money_val
                commission_val = abs(money_val) * commission
                result_money -= commission_val
                commission_money += commission_val
                result_instr = 0.0
                trades_count += 1
            if (result_money > 0):
                # Выход в LONG
                result_money /= (commission + 1.0)
                result_instr = result_money / row['Close']
                commission_money += result_money * commission
                result_money = 0.0
                trades_count += 1
        elif row['Signal'] == -1 and result_instr >= 0:
            # Выход в SHORT если включен режим, иначе в CASHE
            if (result_instr > 0):
                # Выход в CASHE
                money_val = result_instr * row['Close']
                result_money += money_val
                commission_val = abs(money_val) * commission
                result_money -= commission_val
                commission_money += commission_val
                result_instr = 0.0
                trades_count += 1
            if (is_short_enable == True):
                result_money /= (commission + 1.0)
                result_instr = -result_money / row['Close']
                commission_money += result_money * commission
                result_money *= 2.0
                trades_count += 1
        elif row['Signal'] == 0 and result_instr != 0:
            # Выход в CASHE
            money_val = result_instr * row['Close']
            result_money += money_val
            commission_val = abs(money_val) * commission
            result_money -= commission_val
            commission_money += commission_val
            result_instr = 0.0
            trades_count += 1
    if result_instr != 0: # Завершение торговли
        # Выход в CASHE
        money_val = result_instr * strategy_data['Close'].iloc[-1]
        result_money += money_val
        commission_val = abs(money_val) * commission
        result_money -= commission_val
        commission_money += commission_val
        result_instr = 0.0
        trades_count += 1
    return {
        'money': result_money,
        'trades': trades_count,
        'commission': commission_money,
        'money1time': (result_money ** (1/len(strategy_data))) - 1,
        'money1trade': (result_money ** (1/max(trades_count, 1))) - 1
    }

# Добавление дополнительных признаков
def add_technical_indicators(df, sta_wnd_size = 5, ind_window_size = 14):
    df['Average'] = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4
    df['Avg_pct'] = df['Average'].pct_change()  # Отношение к предыдущему
    df['Open_pct'] = df['Open'].pct_change()  # Отношение к предыдущему
    df['Close_pct'] = df['Close'].pct_change()  # Отношение к предыдущему
    df['High_pct'] = df['High'].pct_change()  # Отношение к предыдущему
    df['Low_pct'] = df['Low'].pct_change()  # Отношение к предыдущему
    df['Multi1'] = df['Open'] *  df['Close'] # Умножение 1
    df['Multi2'] = df['High'] *  df['Low'] # Умножение 2
    df['Open2Close'] = df['Open'] / df['Close'] # Деление 1
    df['High2Low'] = df['High'] /  df['Low'] # Деление 2
    df['Avg_sqr'] = df['Average'].apply(lambda x: x**2) # Квадрат

    df['Rolling_Mean'] = df['Average'].rolling(window=sta_wnd_size).mean()  # Скользящее среднее
    df['Rolling_Std'] = df['Average'].rolling(window=sta_wnd_size).std()  # Волатильность

    # Лаги
    for lag in range(1, sta_wnd_size + 1):
        df[f'Lag_Close_{lag}'] = df['Close'].shift(lag)
        df[f'Lag_Open_{lag}'] = df['Open'].shift(lag)
        df[f'Lag_High_{lag}'] = df['High'].shift(lag)
        df[f'Lag_Low_{lag}'] = df['Low'].shift(lag)

    # Дифференциалы
    df['Diff_Rolling_Mean'] = df['Close'] - df['Rolling_Mean']
    df['Diff_High_Low'] = df['High'] - df['Low']

    # Технические индикаторы
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=ind_window_size)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=ind_window_size)
    ema_fast = talib.EMA(df['Close'], timeperiod=int(ind_window_size * 0.25))  # Быстрая EMA
    ema_slow = talib.EMA(df['Close'], timeperiod=ind_window_size)  # Медленная EMA
    df['MACD'] = ema_fast - ema_slow
    df['Signal_Line'] = talib.EMA(df['MACD'], timeperiod=int(ind_window_size * 0.15))

    return df.dropna()

# Загрузка данных
#df = pd.read_csv('Data\\AAPL_data.csv', index_col='Date', parse_dates=True)
df = yf.download("AAPL", interval='1d')
df = df.reset_index()
df.columns = df.columns.droplevel(level=1)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Нормализация данных не используется, т. к. это может ухудшить результат
#scaler = MinMaxScaler()
#df_scaled = scaler.fit_transform(df)

# Формирование признака предсказания для стратегии, основанной на использовании GAP,
# как отношения цены открытия завтра к цене закрытия сегодня
df["Predict"] = df['Open'].shift(-1) / df["Close"]
df = df[:-1] # Удаление последней строки, где в Predict есть NaN

# Добавление дополнительных признаков
df_with_indicators = add_technical_indicators(df)

# Разделение данных на тренировочный и тестовый наборы
X = df_with_indicators.drop('Predict', axis=1)
y = df_with_indicators['Predict']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Линейная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Случайный лес
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Градиентный бустинг Microsoft
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)

# Градиентный бустинг
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Оценка точности моделей
def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)
    m_evs = evs(y_val, predictions)
    print(f'RMSE: {rmse:.4f}, R2: {r2:.4f}, EVS: {m_evs:.4f}')
    return rmse, r2, m_evs

print("Linear Regression:")
rmse_lr, r2_lr, evs_lr = evaluate_model(lr_model, X_val, y_val)

print("\nRandom Forest:")
rmse_rf, r2_rf, evs_rf = evaluate_model(rf_model, X_val, y_val)

print("\nLightGBM:")
rmse_lgbm, r2_lgbm, evs_lgbm = evaluate_model(lgbm_model, X_val, y_val)

print("\nXGBoost:")
rmse_xgb, r2_xgb, evs_xgb = evaluate_model(xgb_model, X_val, y_val)

# Проверка стратегии торговли на GAP с использованием LightGBM, как самого точного метода регрессии
lgbm_df = X_val.copy()
lgbm_df['LGBM'] = lgbm_model.predict(X_val)
# Инициализация сигналов
lgbm_df['Signal'] = 0
# Логика сигналов
lgbm_df.loc[lgbm_df['LGBM'] > 1, 'Signal'] = 1
lgbm_df.loc[lgbm_df['LGBM'] < 1, 'Signal'] = -1
# Проверка результата торговли
lgbm_result = TradingSimple(lgbm_df, commission=0.002)
print("\nLightGBM. Результат работы GAP-стратегии:")
print(f"LightGBM. Результат ДС: {lgbm_result['money']}")
print(f"LightGBM. Количество сделок: {lgbm_result['trades']}")
print(f"LightGBM. Доходность на одну сделку: {lgbm_result['money1trade']}")
print(f"LightGBM. Доходность на единицу времени: {lgbm_result['money1time']}")

# Создание дашборда
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Торговая стратегия'),
    dcc.Graph(id='price-graph'),
    html.H2('Метрики эффективности'),
    html.Table([
        html.Tr([html.Th('Модель'), html.Th('RMSE'), html.Th('R2'), html.Th('EVS')]),
        html.Tr([html.Td('Linear Regression'), html.Td(f'{rmse_lr:.4f}'), html.Td(f'{r2_lr:.4f}'), html.Td(f'{evs_lr:.4f}')]),
        html.Tr([html.Td('Random Forest'), html.Td(f'{rmse_rf:.4f}'), html.Td(f'{r2_rf:.4f}'), html.Td(f'{evs_rf:.4f}')]),
        html.Tr([html.Td('LightGBM'), html.Td(f'{rmse_lgbm:.4f}'), html.Td(f'{r2_lgbm:.4f}'), html.Td(f'{evs_lgbm:.4f}')]),
        html.Tr([html.Td('XGBoost'), html.Td(f'{rmse_xgb:.4f}'), html.Td(f'{r2_xgb:.4f}'), html.Td(f'{evs_xgb:.4f}')])
    ])
])

#if __name__ == '__main__':
#    app.run_server(debug=True)

# Заключение:
# Созданы четыре модели машинного обучения для прогнозирования (регрессии)
# отношения цены открытия завтра к цене закрытия сегодня, т. е. GAP.
# Проведено их тестирование на валидационных данных и оценка их эффективности
# с помощью метрик RMSE, R2, EVS. Наилучшим оказался градиентный бустинг LightGBM.
# Проверена работа торговой стратегии c использованием LightGBM на основе GAP,
# которая показала убыточный результат.
# Также был создан простой дашборд для визуализации результатов.
# Этот подход позволяет сравнивать различные торговые стратегии и
# выбирать наиболее подходящую для конкретных условий рынка.
