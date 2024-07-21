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
        df.set_index('Open time', inplace=True)

        # Ensure that the dataframe has the required columns for mplfinance
        # The columns should typically be 'Open', 'High', 'Low', 'Close', and optionally 'Volume'

        # Создание графика японских свечей
        fig, ax = mpf.plot(df, type='candle', style='charles', returnfig=True, figscale=1.5, figsize=(60, 10))

        plt.show()
        pass


if __name__ == '__main__':
    checker = CheckPreparedData()
    checker.make_plot_data()
