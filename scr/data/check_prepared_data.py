from scr.data.data_preparation import DataPreparation
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

class CheckPreparedData():
    def __init__(self):
        pass

    def make_plot_data(self):
        get_data = DataPreparation(load_new_data=False).prepare_data_main()
        df = get_data.copy()
        df['Date'] = pd.to_datetime(df['Open time'])
        df['Volume'] = df['Quote asset volume']
        df.set_index('Open time', inplace=True)

        def _convect_target(x):
            if x > 0:
                return 1
            if x == 0:
                return 0
            if x < 0:
                return -1

        df['Forecast'] = df['target_predict'].apply(_convect_target)

        # Создаем столбцы для сигналов вверх и вниз и заполняем NaN
        df['Up Signal'] = float('nan')
        df['Down Signal'] = float('nan')

        # Заполняем столбцы значениями сигналов
        df.loc[df['Forecast'] == 1, 'Up Signal'] = df['High']
        df.loc[df['Forecast'] == -1, 'Down Signal'] = df['Low']

        # Проверяем, содержат ли столбцы ненулевые значения
        up_signals = df['Up Signal'].dropna().values
        down_signals = df['Down Signal'].dropna().values

        apdict = []
        if len(up_signals) > 0:
            apdict.append(mpf.make_addplot(df['Up Signal'], type='scatter', marker='^', markersize=100, color='g', panel=0))
        if len(down_signals) > 0:
            apdict.append(mpf.make_addplot(df['Down Signal'], type='scatter', marker='v', markersize=100, color='r', panel=0))

        # Создание графика японских свечей
        fig, ax = mpf.plot(df, type='candle', style='charles', addplot=apdict, volume=True, returnfig=True, figscale=1.5, figsize=(60, 10))

        plt.show()

if __name__ == '__main__':
    checker = CheckPreparedData()
    checker.make_plot_data()
