from binance.client import Client
import keys_log
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
# import mplfinance as mpf
# %pip install python-binance
import os
import pickle

class DataPreparation:
    def __init__(self,
                 load_new_data=False,
                 count_day = 1300,
                 pair_names = "ETHUSDT",
                 interval = 'd'):
        self.load_new_data = load_new_data
        self.count_day = count_day
        self.pair_names = pair_names
        self.interval = interval


    def set_work_path(self):
        # Получаем текущий рабочий каталог
        current_directory = os.getcwd()

        # Ищем позицию, где находится "su-data-platform" в пути
        index = current_directory.rfind("search_anomaly_Bitcoin")

        # Если "su-data-platform" найден, переходим к его родительскому каталогу
        if index != -1:
            project_directory = current_directory[:index + len("search_anomaly_Bitcoin")]

            # Переходим в родительский каталог "su-data-platform"
            os.chdir(project_directory)

        current_directory = os.getcwd()
        print('Рабочая дериктория:', {current_directory})

    def get_start_end_date(self, count_day):
        now = datetime.now()
        year_ago = now - timedelta(days=count_day)

        date_start = year_ago.strftime('%d.%m.%Y')
        date_end = now.strftime('%d.%m.%Y')

        return date_start, date_end
    def get_data_trading(self):
        '''
        Получаю данные торговле: Цены по дням, объем, кол-во сделов. (прочие).
        Данные беру с binance потому что это наиболее ликвидная биржа.
        Возможно получение разных вариантов данных. В зависимости от параметров.
        :return:

        self.df_quotes
        'Quote asset volume' - объем торгов за день котируемого актива.
        'quote asset' - Котируемый актив - Основной актив, к примеру USDT
        'base asset' - Базовый актив - что мы оцениваем с помощью котируемого.
        'Taker buy base asset volume' - покупатели исполняют ордера на покупку базового актива.
        'Taker buy quote asset volume' - покупатели исполняют ордера на покупку котируемого актива.
        'Number of trades' - кол-во совершонных сделок.ф

        '''

        date_start, date_end = self.get_start_end_date(self.count_day)

        client = Client(keys_log.api_key, keys_log.api_secret)
        dict_interval = {'d': Client.KLINE_INTERVAL_1DAY}

        load_list = client.get_historical_klines(self.pair_names, dict_interval[self.interval.lower()], date_start, date_end)
        self.df_quotes = pd.DataFrame(load_list, columns=['Open time',
                                                     'Open',
                                                     'High',
                                                     'Low',
                                                     'Close',
                                                     'Volume',
                                                     'Close time',
                                                     'Quote asset volume',
                                                     'Number of trades',
                                                     'Taker buy base asset volume',
                                                     'Taker buy quote asset volume',
                                                     'Ignore'])

        name = f'{self.pair_names}_{self.count_day}d_interval-{self.interval}'
        file_path = f'data/raw/{name}.pkl'

        info_save = {"count_day":self.count_day, "interval":self.interval, "pair_names":self.pair_names}
        with open(file_path, 'wb') as file:
            pickle.dump({'df':self.df_quotes, 'info':info_save}, file)

    def _read_save_quotes(self):
        '''

        :return:
        '''
        name = f'{self.pair_names}_{self.count_day}d_interval-{self.interval}'
        file_path = f'data/raw/{name}.pkl'

        # Чтение объекта из файла формата pickle
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)

        # Извлечение DataFrame и дополнительной информации
        self.df_quotes = loaded_data['df']
        self.info_save = loaded_data['info']

    def _convert_to_float(self):
        def convert_formats(df):
            df = df.copy()
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
            col_new_type = {'Open': float,
                            'High': float,
                            'Low': float,
                            'Close': float,
                            'Volume': float,
                            'Quote asset volume': float,
                            'Number of trades': int,
                            'Taker buy base asset volume': float,
                            'Taker buy quote asset volume': float
                            }

            df = df.astype(col_new_type)

            return df

        self.df_quotes = convert_formats(self.df_quotes)
        # self.df_quotes = self.df_quotes.set_index('Open time')


    def _convert_to_diff_format(self):
        self.df_quotes.sort_values(by=['Open time'], inplace=True)

        self.df_quotes_diff = self.df_quotes[['Open', 'High', "Low", "Close", "Volume", "Quote asset volume", "Number of trades"]].pct_change()
        self.df_quotes_diff.to_excel('self.df_quotes_diff.xlsx', index=False)
        self.df_quotes.to_excel('self.df_quotes.xlsx', index=False)
        pass


    def prepare_data_main(self):
        self.set_work_path()
        if self.load_new_data:
            self.get_data_trading()
        else:
            self._read_save_quotes()
        self._convert_to_float()

        self._convert_to_diff_format()
        pass


if __name__ == '__main__':
    DataPreparation(load_new_data=False).prepare_data_main()