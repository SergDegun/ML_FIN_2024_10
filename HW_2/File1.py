import pandas as pd
#import yfinance as yf
import talib
import streamlit as st
from sklearn.model_selection import train_test_split
#from ta.trend import SMAIndicator, EMAIndicator, MACD
#from ta.volatility import BollingerBands
#from ta.momentum import RSIIndicator

# Простая реализация процесса торговли для стратегии
def TradingSimple(strategy_data, is_enable_short = False):
    result_money = 1
    result_instr = 0
    trades_count = 0
    # Проход по всем позициям
    for index, row in strategy_data.iterrows():
        if row['Signal'] == 1 and result_money != 0:
            # Выход в LONG
            result_instr = result_money / row['close']
            result_money = 0
            trades_count += 1
            #print("buy:" + str(result_instr))
        elif row['Signal'] == -1 and result_instr != 0:
            # Выход в SHORT если включен, иначе в CASHE
            result_money = result_instr * row['close']
            result_instr = 0
            trades_count += 1
            #print("sell:" + str(result_money))
        elif row['Signal'] == 0 and result_instr != 0:
            # Выход в CASHE
            result_money += result_instr * row['close']
            result_instr = 0
            trades_count += 1
            # print("sell:" + str(result_money))
    # Проверка завершения торговли
    if result_money == 0:
        result_money = result_instr * strategy_data['close'].iloc[-1]
        result_instr = 0
    return result_money, trades_count

# Проверка стратегии MACD
def CheckMACD(X_train_val, fastperiod, slowperiod, signalperiod):
    macd_df = X_train_val.copy()
    macd_df['MACD'], macd_df['MACD_Signal'], macd_df['MACD_Hist'] = talib.MACD(
        macd_df["close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    # Инициализация сигналов
    macd_df['Signal'] = 0
    # Логика сигналов
    macd_df.loc[macd_df['MACD'] > macd_df['MACD_Signal'], 'Signal'] = 1
    macd_df.loc[macd_df['MACD'] < macd_df['MACD_Signal'], 'Signal'] = -1
    # Проверка результата торговли
    return TradingSimple(macd_df)

# Проверка стратегии STOCH
def CheckSTOCH(X_train_val, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype):
    stoch_df = X_train_val.copy()
    # Расчет стохастического осциллятора
    stoch_df['slowk'], stoch_df['slowd'] = talib.STOCH(
        stoch_df['high'], stoch_df['low'], stoch_df['close'],
        fastk_period=fastk_period, slowk_period=slowk_period,
        slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)
    # Инициализация сигналов
    stoch_df['Signal'] = 0
    # Логика для генерации сигналов
    stoch_df.loc[stoch_df['slowk'] > stoch_df['slowd'], 'Signal'] = 1
    stoch_df.loc[stoch_df['slowk'] < stoch_df['slowd'], 'Signal'] = -1
    # Проверка результата торговли
    return TradingSimple(stoch_df)


# Проверка стратегии TEMA
def CheckTEMA(X_train_val, period):
    tema_df = X_train_val.copy()
    # EMA1: Первоначальная EMA от цены
    tema_df['EMA1'] = talib.EMA(tema_df['close'], timeperiod=period)
    # EMA2: EMA от EMA1
    tema_df['EMA2'] = talib.EMA(tema_df['EMA1'], timeperiod=period)
    # EMA3: EMA от EMA2
    tema_df['EMA3'] = talib.EMA(tema_df['EMA2'], timeperiod=period)
    # Triple EMA
    tema_df['TripleEMA'] = (3 * tema_df['EMA1']) - (3 * tema_df['EMA2']) + tema_df['EMA3']
    tema_df = tema_df.dropna()
    # Инициализация сигналов
    tema_df['Signal'] = 0
    # Логика для входа в короткую позицию (Short)
    tema_df.loc[tema_df['close'] < tema_df['TripleEMA'], 'Signal'] = -1
    # Логика для входа в длинную позицию (Long)
    tema_df.loc[tema_df['close'] > tema_df['TripleEMA'], 'Signal'] = 1
    # Проверка результата торговли
    return TradingSimple(tema_df)


# Проверка стратегии BBOL
def CheckBBOL(X_train_val, period):
    bb_df = X_train_val.copy()
    # Вычисляем среднюю линию (SMA)
    bb_df['MA'] = talib.SMA(bb_df['close'], timeperiod=period)
    # Вычисляем стандартное отклонение
    bb_df['STD'] = bb_df['close'].rolling(window=period).std()
    # Вычисляем верхние и нижние полосы для разных уровней стандартного отклонения
    bb_df['UpperBB_1SD'] = bb_df['MA'] + (1 * bb_df['STD'])
    bb_df['UpperBB_2SD'] = bb_df['MA'] + (2 * bb_df['STD'])
    bb_df['LowerBB_1SD'] = bb_df['MA'] - (1 * bb_df['STD'])
    bb_df['LowerBB_2SD'] = bb_df['MA'] - (2 * bb_df['STD'])
    # Инициализация сигналов
    bb_df['Signal'] = 0
    # Вход в позицию BUY
    bb_df.loc[bb_df['close'] > bb_df['UpperBB_1SD'], 'Signal'] = 1
    # Вход в позицию SELL
    bb_df.loc[bb_df['close'] < bb_df['LowerBB_1SD'], 'Signal'] = -1
    # Проверка результата торговли
    return TradingSimple(bb_df)


# Проверка стратегии PATR
def CheckPATR(X_train_val):
    candlestick_df = X_train_val.copy()
    # Распознавание паттерна поглощения
    candlestick_df['engulfing'] = talib.CDLENGULFING(candlestick_df['open'], candlestick_df['high'],
                                                     candlestick_df['low'], candlestick_df['close'])
    # Инициализация столбца сигналов нулями
    candlestick_df['Signal'] = 0
    # Сигналы на покупку (бычье поглощение)
    candlestick_df.loc[candlestick_df['engulfing'] > 0, 'Signal'] = 1
    # Сигналы на продажу (медвежье поглощение)
    candlestick_df.loc[candlestick_df['engulfing'] < 0, 'Signal'] = -1
    # Проверка результата торговли
    return TradingSimple(candlestick_df)


#################################################################
# Загрузка данных и очистка данных
#data = yf.Ticker("AAPL").history(period="max")
data = pd.read_csv('Data\\AAPL.csv')
data.columns = data.columns.str.lower()
data = data.dropna()
data.index = pd.to_datetime(data.index)
data = data.drop(['dividends'], axis=1)     # Удаление Dividends
data = data.drop(['stock splits'], axis=1)  # Удаление Stock Splits

# Разделение данных на тренировочную и тестовую выборки
X_train_val, X_test, y_train_val, y_test = train_test_split(data, data['close'], test_size=0.2, shuffle=False)
# Разделение тренировочной выборки на тренировочную и валидационную
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=False)


#################################################################
# Проверка стратегии MACD
#result_money, trades_count = CheckMACD(X_train_val, fastperiod=12, slowperiod=26, signalperiod=9)
#print(f"MACD. Итоговый результат ДС: {result_money}")
#print(f"MACD. Итоговое количество сделок: {trades_count}")

# Поиск гиперпараметров для стратегии MACD
MACD_optim = True
if (MACD_optim):
    print(f"###############################################")
    print(f"MACD. Поиск гиперпараметров для стратегии")
    macd_best_money = 0
    macd_best_trades = 0
    macd_best_fastperiod = 0
    macd_best_slowperiod = 0
    macd_best_signalperiod = 0
    # Подбор гиперпараметров
    for fastperiod in (2, 7, 12):
        for slowperiod in (12, 19, 26):
            for signalperiod in (3, 6, 9, 12):
                # Проверка результата торговли
                result_money, trades_count = CheckMACD(X_train_val,
                    fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                if (result_money > macd_best_money):
                    macd_best_money = result_money
                    macd_best_trades = trades_count
                    macd_best_fastperiod = fastperiod
                    macd_best_slowperiod = slowperiod
                    macd_best_signalperiod = signalperiod
    print(f"MACD. Лучший результат ДС: {macd_best_money}")
    print(f"MACD. Лучшее количество сделок: {macd_best_trades}")
    print(f"MACD. Доходность на единицу времени: {macd_best_money / len(X_train_val)}")
    print(f"MACD. Лучшее значение fastperiod: {macd_best_fastperiod}")
    print(f"MACD. Лучшее значение slowperiod: {macd_best_slowperiod}")
    print(f"MACD. Лучшее значение signalperiod: {macd_best_signalperiod}")


#################################################################
# Проверка стратегии STOCH
#result_money, trades_count = CheckSTOCH(X_train_val, fastk_period=14, slowk_period=7, slowk_matype=0, slowd_period=7, slowd_matype=0)
# Вывод результатов
#print(f"STOCH. Итоговый результат ДС: {result_money}")
#print(f"STOCH. Итоговое количество сделок: {trades_count}")

# Поиск гиперпараметров для стратегии STOCH
STOCH_optim = True
if (STOCH_optim):
    print(f"###############################################")
    print(f"STOCH. Поиск гиперпараметров для стратегии")
    stoch_best_money = 0
    stoch_best_trades = 0
    stoch_best_fastk_period = 0
    stoch_best_slowk_period = 0
    stoch_best_slowd_period = 0
    # Подбор гиперпараметров
    for fastk_period in (10, 14, 18):
        for slowk_period in (4, 7, 10):
            for slowd_period in (4, 7, 10):
                # Проверка результата торговли
                result_money, trades_count = CheckSTOCH(X_train_val,
                    fastk_period=fastk_period, slowk_period=slowk_period,
                    slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)
                if (result_money > stoch_best_money):
                    stoch_best_money = result_money
                    stoch_best_trades = trades_count
                    stoch_best_fastk_period = fastk_period
                    stoch_best_slowk_period = slowk_period
                    stoch_best_slowd_period = slowd_period
    print(f"STOCH. Лучший результат ДС: {stoch_best_money}")
    print(f"STOCH. Лучшее количество сделок: {stoch_best_trades}")
    print(f"STOCH. Доходность на единицу времени: {stoch_best_money / len(X_train_val)}")
    print(f"STOCH. Лучшее значение fastk_period: {stoch_best_fastk_period}")
    print(f"STOCH. Лучшее значение slowk_period: {stoch_best_slowk_period}")
    print(f"STOCH. Лучшее значение slowd_period: {stoch_best_slowd_period}")


#################################################################
# Проверка стратегии TEMA
#result_money, trades_count = CheckTEMA(X_train_val,  period=55)
# Вывод результатов
#print(f"TEMA. Итоговый результат ДС: {result_money}")
#print(f"TEMA. Итоговое количество сделок: {trades_count}")

# Поиск гиперпараметров для стратегии TEMA
TEMA_optim = True
if (TEMA_optim):
    print(f"###############################################")
    print(f"TEMA. Поиск гиперпараметров для стратегии")
    tema_best_money = 0
    tema_best_trades = 0
    tema_best_period = 0
    # Подбор гиперпараметров
    for period in range(2, 100, 1):
        # Проверка результата торговли
        result_money, trades_count = CheckTEMA(X_train_val, period)
        if (result_money > tema_best_money):
            tema_best_money = result_money
            tema_best_trades = trades_count
            tema_best_period = period
    print(f"TEMA. Лучший результат ДС: {tema_best_money}")
    print(f"TEMA. Лучшее количество сделок: {tema_best_trades}")
    print(f"TEMA. Доходность на единицу времени: {tema_best_money / len(X_train_val)}")
    print(f"TEMA. Лучшее значение period: {tema_best_period}")


#################################################################
# Проверка стратегии с использованием линий Болинджера BBOL
#result_money, trades_count = CheckBBOL(X_train_val,  period=55)
# Вывод результатов
#print(f"BBOL. Итоговый результат ДС: {result_money}")
#print(f"BBOL. Итоговое количество сделок: {trades_count}")

# Поиск гиперпараметров для стратегии BBOL
BBOL_optim = True
if (BBOL_optim):
    print(f"###############################################")
    print(f"BBOL. Поиск гиперпараметров для стратегии")
    bbol_best_money = 0
    bbol_best_trades = 0
    bbol_best_period = 0
    # Подбор гиперпараметров
    for period in range(2, 100, 1):
        # Проверка результата торговли
        result_money, trades_count = CheckBBOL(X_train_val, period)
        if (result_money >  bbol_best_money):
            bbol_best_money = result_money
            bbol_best_trades = trades_count
            bbol_best_period = period
    print(f"BBOL. Лучший результат ДС: {bbol_best_money}")
    print(f"BBOL. Лучшее количество сделок: {bbol_best_trades}")
    print(f"BBOL. Доходность на единицу времени: {bbol_best_money / len(X_train_val)}")
    print(f"BBOL. Лучшее значение period: {bbol_best_period}")


#################################################################
# Проверка стратегии с использованием паттернов свечей
#result_money, trades_count = CheckPATR(X_train_val)
# Вывод результатов
#print(f"PATR. Итоговый результат ДС: {result_money}")
#print(f"PATR. Итоговое количество сделок: {trades_count}")


#################################################################
print(f"###############################################")
print("Тестирование стратегий на тестовой выборке")

# Проверка стратегии MACD
result_money, trades_count = CheckMACD(X_test, fastperiod=macd_best_fastperiod, slowperiod=macd_best_slowperiod,
                                       signalperiod=macd_best_signalperiod)
print(f"MACD. Тестовый результат ДС: {result_money}")
print(f"MACD. Тестовое количество сделок: {trades_count}")
print(f"MACD. Доходность на единицу времени: {result_money / len(X_test)}")
print(f"------------------------------------------")
# Проверка стратегии STOCH
result_money, trades_count = CheckSTOCH(X_test, fastk_period=stoch_best_fastk_period, slowk_period=stoch_best_slowk_period,
                                        slowk_matype=0, slowd_period=stoch_best_slowd_period, slowd_matype=0)
# Вывод результатов
print(f"STOCH. Тестовый результат ДС: {result_money}")
print(f"STOCH. Тестовое количество сделок: {trades_count}")
print(f"STOCH. Доходность на единицу времени: {result_money / len(X_test)}")
print(f"------------------------------------------")
# Проверка стратегии TEMA
result_money, trades_count = CheckTEMA(X_test,  period=tema_best_period)
# Вывод результатов
print(f"TEMA. Тестовый результат ДС: {result_money}")
print(f"TEMA. Тестовое количество сделок: {trades_count}")
print(f"TEMA. Доходность на единицу времени: {result_money / len(X_test)}")
print(f"------------------------------------------")
# Проверка стратегии с использованием линий Болинджера BBOL
result_money, trades_count = CheckBBOL(X_test,  period=bbol_best_period)
# Вывод результатов
print(f"BBOL. Тестовый результат ДС: {result_money}")
print(f"BBOL. Тестовое количество сделок: {trades_count}")
print(f"BBOL. Доходность на единицу времени: {result_money / len(X_test)}")
print(f"------------------------------------------")
# Проверка стратегии с использованием паттернов свечей
result_money, trades_count = CheckPATR(X_test)
# Вывод результатов
print(f"PATR. Тестовый результат ДС: {result_money}")
print(f"PATR. Тестовое количество сделок: {trades_count}")
print(f"PATR. Доходность на единицу времени: {result_money / len(X_test)}")
print(f"------------------------------------------")


# ВЫВОДЫ: на обучающей выборке можно подобрать такие гиперпараметры стратегий, что они будут иметь очень высокую доходность.
#         Но на тестовой выборке эти стратегии с этими оптимальными гиперпараметрами, скорее всего, будут иметь доходность
#         значительно меньшую, чем на обучающей.
