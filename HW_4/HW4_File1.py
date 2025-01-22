import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Загрузка данных
df = pd.read_csv('Data\\AAPL.csv', index_col='Date', parse_dates=True)

# Нормализация данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Добавление технических индикаторов
def add_technical_indicators(df):
    # Скользящая средняя за 20 дней
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Индекс относительной силы (RSI)
    delta = df['Close'].diff().dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(span=14).mean()
    roll_down = down.abs().ewm(span=14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    return df.dropna()


df_with_indicators = add_technical_indicators(df)

# Разделение данных на тренировочный и тестовый наборы
X = scaled_data[:-1]
y = scaled_data[1:, 3]  # Прогнозируем цену закрытия

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Линейная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Случайный лес
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Градиентный бустинг
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# Оценка точности моделей
def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    print(f'MSE: {mse:.4f}, R2: {r2:.4f}')

print("Linear Regression:")
evaluate_model(lr_model, X_val, y_val)

print("\nRandom Forest:")
evaluate_model(rf_model, X_val, y_val)

print("\nLightGBM:")
evaluate_model(lgb_model, X_val, y_val)

# Создание дашборда
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Торговая стратегия'),
    dcc.Graph(id='price-graph'),
    html.H2('Метрики эффективности'),
    html.Table([
        html.Tr([html.Th('Модель'), html.Th('MSE'), html.Th('R2')]),
        html.Tr([html.Td('Linear Regression'), html.Td(f'{mse_lr:.4f}'), html.Td(f'{r2_lr:.4f}')]),
        html.Tr([html.Td('Random Forest'), html.Td(f'{mse_rf:.4f}'), html.Td(f'{r2_rf:.4f}')]),
        html.Tr([html.Td('LightGBM'), html.Td(f'{mse_lgb:.4f}'), html.Td(f'{r2_lgb:.4f}')])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)

# Заключение:
# Мы создали три модели машинного обучения для прогнозирования цены актива,
# провели их тестирование на валидационных данных и оценили их эффективность
# с помощью метрик MSE и R2. Также был создан простой дашборд для визуализации результатов.
# Этот подход позволяет сравнивать различные торговые стратегии и
# выбирать наиболее подходящую для конкретных условий рынка.
