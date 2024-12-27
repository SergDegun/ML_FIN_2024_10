import pandas as pd
import yfinance as yf
import talib
import itertools
import warnings
#import backtesting
#from backtesting import Backtest, Strategy

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
                money_val = result_instr * row['close']
                result_money += money_val
                commission_val = abs(money_val) * commission
                result_money -= commission_val
                commission_money += commission_val
                result_instr = 0.0
                trades_count += 1
            if (result_money > 0):
                # Выход в LONG
                result_money /= (commission + 1.0)
                result_instr = result_money / row['close']
                commission_money += result_money * commission
                result_money = 0.0
                trades_count += 1
        elif row['Signal'] == -1 and result_instr >= 0:
            # Выход в SHORT если включен режим, иначе в CASHE
            if (result_instr > 0):
                # Выход в CASHE
                money_val = result_instr * row['close']
                result_money += money_val
                commission_val = abs(money_val) * commission
                result_money -= commission_val
                commission_money += commission_val
                result_instr = 0.0
                trades_count += 1
            if (is_short_enable == True):
                result_money /= (commission + 1.0)
                result_instr = -result_money / row['close']
                commission_money += result_money * commission
                result_money *= 2.0
                trades_count += 1
        elif row['Signal'] == 0 and result_instr != 0:
            # Выход в CASHE
            money_val = result_instr * row['close']
            result_money += money_val
            commission_val = abs(money_val) * commission
            result_money -= commission_val
            commission_money += commission_val
            result_instr = 0.0
            trades_count += 1
    if result_instr != 0: # Завершение торговли
        # Выход в CASHE
        money_val = result_instr * strategy_data['close'].iloc[-1]
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
    return TradingSimple(macd_df, commission=0.002)

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
    return TradingSimple(stoch_df, commission=0.002)


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
    return TradingSimple(tema_df, commission=0.002)


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
    return TradingSimple(bb_df, commission=0.002)


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
    return TradingSimple(candlestick_df, commission=0.002)


#################################################################
# Загрузка данных и подготовка (очистка) данных
#data = pd.read_csv('Data\\AAPL.csv')
data = yf.download("AAPL", interval='1d')
data.columns = data.columns.droplevel(level=1)
data.columns = data.columns.str.lower()
data = data.dropna()
data.index = pd.to_datetime(data.index)

# Разделение данных на тренировочную\валидационную и тестовую выборки относительно даты 01.03.2024
split_date = pd.to_datetime('2024-03-01')
X_train_val = data[data.index < split_date]
X_test = data[data.index >= split_date]


#################################################################
# Проверка стратегии MACD
#result = CheckMACD(X_train_val, fastperiod=12, slowperiod=26, signalperiod=9)
#print(f"MACD. Итоговый результат ДС: {result['money']}")
#print(f"MACD. Итоговое количество сделок: {result['trades']}")

# Поиск гиперпараметров для стратегии MACD
MACD_optim = True
if (MACD_optim):
    print(f"###############################################")
    print(f"MACD. Поиск гиперпараметров для стратегии")
    macd_best_result = {'money': 0}
    macd_best_fastperiod = 0
    macd_best_slowperiod = 0
    macd_best_signalperiod = 0
    # Подбор гиперпараметров
    fastperiod_list = [2, 7, 12]
    slowperiod_list = [12, 19, 26]
    signalperiod_list = [3, 6, 9, 12]
    for fastperiod, slowperiod, signalperiod in itertools.product(fastperiod_list, slowperiod_list, signalperiod_list):
        # Проверка результата торговли
        result = CheckMACD(X_train_val,
            fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        if (macd_best_result['money'] < result['money']):
            macd_best_result = result
            macd_best_fastperiod = fastperiod
            macd_best_slowperiod = slowperiod
            macd_best_signalperiod = signalperiod
    print(f"MACD. Лучший результат ДС: {macd_best_result['money']}")
    print(f"MACD. Лучшее количество сделок: {macd_best_result['trades']}")
    print(f"MACD. Доходность на одну сделку: {macd_best_result['money1trade']}")
    print(f"MACD. Доходность на единицу времени: {macd_best_result['money1time']}")
    print(f"MACD. Лучшее значение fastperiod: {macd_best_fastperiod}")
    print(f"MACD. Лучшее значение slowperiod: {macd_best_slowperiod}")
    print(f"MACD. Лучшее значение signalperiod: {macd_best_signalperiod}")


#################################################################
# Проверка стратегии STOCH
#result = CheckSTOCH(X_train_val, fastk_period=14, slowk_period=7, slowk_matype=0, slowd_period=7, slowd_matype=0)
# Вывод результатов
#print(f"STOCH. Итоговый результат ДС: {result['money']}")
#print(f"STOCH. Итоговое количество сделок: {result['trades']}")

# Поиск гиперпараметров для стратегии STOCH
STOCH_optim = True
if (STOCH_optim):
    print(f"###############################################")
    print(f"STOCH. Поиск гиперпараметров для стратегии")
    stoch_best_result =  {'money': 0}
    stoch_best_fastk_period = 0
    stoch_best_slowk_period = 0
    stoch_best_slowd_period = 0
    # Подбор гиперпараметров
    fastk_period_list = [10, 14, 18]
    slowk_period_list = [4, 7, 10]
    slowd_period_list = [4, 7, 10]
    for fastk_period, slowk_period, slowd_period in itertools.product(fastk_period_list, slowk_period_list, slowd_period_list):
        # Проверка результата торговли
        result = CheckSTOCH(X_train_val, fastk_period=fastk_period,
            slowk_period=slowk_period, slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)
        if (stoch_best_result['money'] < result['money']):
            stoch_best_result = result
            stoch_best_fastk_period = fastk_period
            stoch_best_slowk_period = slowk_period
            stoch_best_slowd_period = slowd_period
    print(f"STOCH. Лучший результат ДС: {stoch_best_result['money']}")
    print(f"STOCH. Лучшее количество сделок: {stoch_best_result['trades']}")
    print(f"STOCH. Доходность на одну сделку: {stoch_best_result['money1trade']}")
    print(f"STOCH. Доходность на единицу времени: {stoch_best_result['money1time']}")
    print(f"STOCH. Лучшее значение fastk_period: {stoch_best_fastk_period}")
    print(f"STOCH. Лучшее значение slowk_period: {stoch_best_slowk_period}")
    print(f"STOCH. Лучшее значение slowd_period: {stoch_best_slowd_period}")


#################################################################
# Проверка стратегии TEMA
#result = CheckTEMA(X_train_val,  period=55)
# Вывод результатов
#print(f"TEMA. Итоговый результат ДС: {result['money']}")
#print(f"TEMA. Итоговое количество сделок: {result['trades']}")

# Поиск гиперпараметров для стратегии TEMA
TEMA_optim = True
if (TEMA_optim):
    print(f"###############################################")
    print(f"TEMA. Поиск гиперпараметров для стратегии")
    tema_best_result =  {'money': 0}
    tema_best_period = 0
    # Подбор гиперпараметров
    for period in range(2, 100, 1):
        # Проверка результата торговли
        result = CheckTEMA(X_train_val, period)
        if (tema_best_result['money'] < result['money']):
            tema_best_result = result
            tema_best_period = period
    print(f"TEMA. Лучший результат ДС: {tema_best_result['money']}")
    print(f"TEMA. Лучшее количество сделок: {tema_best_result['trades']}")
    print(f"TEMA. Доходность на одну сделку: {tema_best_result['money1trade']}")
    print(f"TEMA. Доходность на единицу времени: {tema_best_result['money1time']}")
    print(f"TEMA. Лучшее значение period: {tema_best_period}")


#################################################################
# Проверка стратегии с использованием линий Болинджера BBOL
#result = CheckBBOL(X_train_val,  period=55)
# Вывод результатов
#print(f"BBOL. Итоговый результат ДС: {result['money']}")
#print(f"BBOL. Итоговое количество сделок: {result['trades']}")

# Поиск гиперпараметров для стратегии BBOL
BBOL_optim = True
if (BBOL_optim):
    print(f"###############################################")
    print(f"BBOL. Поиск гиперпараметров для стратегии")
    bbol_best_result =  {'money': 0}
    bbol_best_period = 0
    # Подбор гиперпараметров
    for period in range(2, 100, 1):
        # Проверка результата торговли
        result = CheckBBOL(X_train_val, period)
        if (bbol_best_result['money'] < result['money']):
            bbol_best_result = result
            bbol_best_period = period
    print(f"BBOL. Лучший результат ДС: {bbol_best_result['money']}")
    print(f"BBOL. Лучшее количество сделок: {bbol_best_result['trades']}")
    print(f"BBOL. Доходность на одну сделку: {bbol_best_result['money1trade']}")
    print(f"BBOL. Доходность на единицу времени: {bbol_best_result['money1time']}")
    print(f"BBOL. Лучшее значение period: {bbol_best_period}")


#################################################################
# Проверка стратегии с использованием паттернов свечей
#result = CheckPATR(X_train_val)
# Вывод результатов
#print(f"PATR. Итоговый результат ДС: {result['money']}")
#print(f"PATR. Итоговое количество сделок: {result['trades']}")


#################################################################
print(f"###############################################")
print("Тестирование стратегий на тестовой выборке")

# Проверка стратегии MACD
result = CheckMACD(X_test, fastperiod=macd_best_fastperiod,
                   slowperiod=macd_best_slowperiod, signalperiod=macd_best_signalperiod)
print(f"MACD. Тестовый результат ДС: {result['money']}")
print(f"MACD. Тестовое количество сделок: {result['trades']}")
print(f"MACD. Доходность на одну сделку: {result['money1trade']}")
print(f"MACD. Доходность на единицу времени: {result['money1time']}")
print(f"------------------------------------------")
# Проверка стратегии STOCH
result = CheckSTOCH(X_test, fastk_period=stoch_best_fastk_period, slowk_period=stoch_best_slowk_period,
                    slowk_matype=0, slowd_period=stoch_best_slowd_period, slowd_matype=0)
# Вывод результатов
print(f"STOCH. Тестовый результат ДС: {result['money']}")
print(f"STOCH. Тестовое количество сделок: {result['trades']}")
print(f"STOCH. Доходность на одну сделку: {result['money1trade']}")
print(f"STOCH. Доходность на единицу времени: {result['money1time']}")
print(f"------------------------------------------")
# Проверка стратегии TEMA
result = CheckTEMA(X_test,  period=tema_best_period)
# Вывод результатов
print(f"TEMA. Тестовый результат ДС: {result['money']}")
print(f"TEMA. Тестовое количество сделок: {result['trades']}")
print(f"TEMA. Доходность на одну сделку: {result['money1trade']}")
print(f"TEMA. Доходность на единицу времени: {result['money1time']}")
print(f"------------------------------------------")
# Проверка стратегии с использованием линий Болинджера BBOL
result = CheckBBOL(X_test,  period=bbol_best_period)
# Вывод результатов
print(f"BBOL. Тестовый результат ДС: {result['money']}")
print(f"BBOL. Тестовое количество сделок: {result['trades']}")
print(f"BBOL. Доходность на одну сделку: {result['money1trade']}")
print(f"BBOL. Доходность на единицу времени: {result['money1time']}")
print(f"------------------------------------------")
# Проверка стратегии с использованием паттернов свечей
result = CheckPATR(X_test)
# Вывод результатов
print(f"PATR. Тестовый результат ДС: {result['money']}")
print(f"PATR. Тестовое количество сделок: {result['trades']}")
print(f"PATR. Доходность на одну сделку: {result['money1trade']}")
print(f"PATR. Доходность на единицу времени: {result['money1time']}")
print(f"------------------------------------------")


# ВЫВОДЫ: 1. на обучающей выборке можно подобрать такие гиперпараметры стратегий, что они будут иметь высокую доходность.
#         Но на тестовой выборке эти стратегии с этими оптимальными гиперпараметрами, скорее всего, будут иметь доходность
#         значительно меньшую, чем на обучающей.
#         2. самой перспективной из рассмотренных стратегий на тестовом участке является STOCH
