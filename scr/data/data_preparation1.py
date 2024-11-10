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
from pathlib import Path


class DataPreparation:
    def __init__(
        self, load_new_data=False, count_day=1300, pair_names="ETHUSDT", interval="d"
    ):
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
            project_directory = current_directory[
                : index + len("search_anomaly_Bitcoin")
            ]

            # Переходим в родительский каталог "su-data-platform"
            os.chdir(project_directory)

        current_directory = os.getcwd()
        print("Рабочая дериктория:", {current_directory})

    def get_start_end_date(self, count_day):
        now = datetime.now()
        year_ago = now - timedelta(days=count_day)

        date_start = year_ago.strftime("%d.%m.%Y")
        date_end = now.strftime("%d.%m.%Y")

        return date_start, date_end

    def get_data_trading(self):
        """
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

        """

        date_start, date_end = self.get_start_end_date(self.count_day)

        client = Client(keys_log.api_key, keys_log.api_secret)
        dict_interval = {"d": Client.KLINE_INTERVAL_1DAY}

        load_list = client.get_historical_klines(
            self.pair_names, dict_interval[self.interval.lower()], date_start, date_end
        )
        self.df_quotes = pd.DataFrame(
            load_list,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote asset volume",
                "Number of trades",
                "Taker buy base asset volume",
                "Taker buy quote asset volume",
                "Ignore",
            ],
        )

        name = f"{self.pair_names}_{self.count_day}d_interval-{self.interval}"
        file_path = f"data/raw/{name}.pkl"

        info_save = {
            "count_day": self.count_day,
            "interval": self.interval,
            "pair_names": self.pair_names,
        }
        with open(file_path, "wb") as file:
            pickle.dump({"df": self.df_quotes, "info": info_save}, file)

    def _read_save_quotes(self):
        """

        :return:
        """
        name = f"{self.pair_names}_{self.count_day}d_interval-{self.interval}"
        file_path = f"data/raw/{name}.pkl"

        # Чтение объекта из файла формата pickle
        with open(file_path, "rb") as file:
            loaded_data = pickle.load(file)

        # Извлечение DataFrame и дополнительной информации
        self.df_quotes = loaded_data["df"]
        self.info_save = loaded_data["info"]

    def _convert_to_float(self):
        def convert_formats(df):
            df = df.copy()
            df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
            df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
            col_new_type = {
                "Open": float,
                "High": float,
                "Low": float,
                "Close": float,
                "Volume": float,
                "Quote asset volume": float,
                "Number of trades": int,
                "Taker buy base asset volume": float,
                "Taker buy quote asset volume": float,
            }

            df = df.astype(col_new_type)

            return df

        self.df_quotes = convert_formats(self.df_quotes)
        # self.df_quotes = self.df_quotes.set_index('Open time')

    def _convert_to_diff_format(self):
        self.df_quotes.sort_values(by=["Open time"], inplace=True)

        self.df_quotes_diff = self.df_quotes[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Quote asset volume",
                "Number of trades",
            ]
        ].pct_change()
        for col in list(self.df_quotes_diff):
            new_col = col + "_diff"
            self.df_quotes[new_col] = self.df_quotes[col]

        self.df_quotes_diff.to_excel("self.df_quotes_diff.xlsx", index=False)
        self.df_quotes.to_excel("self.df_quotes.xlsx", index=False)
        pass

    def _find_target_bar(self, diff_find):
        """
        Нахожу целевые бары
        diff_find - процент изменения цены за бар.

        Итоговая колонка содержит:
        0 - если мзенение меньше diff_find
        фактическое максимальное изменение от цены open до 2х из (high, low)
        цена close - в этом случае не нужна.

        Для создания прогноза нужно будет сместить занную колонку на 1 в низ.

        Постобработка:
        1) Нужно убрать ситуации когда следующий день идёт и в минус и в плюс.
        т.е. когда просто большая волатильность.
        Такие дни я хочу выкинуть потому что меня бы вибило по стопу. (в тамом случае предсказываем волатильность,
        но без превызки к направлению)

        2) Нужно убрать ситуации когда движение происхоидло только из-за ВНЕЗАМНЫХ новостей.
        :return:
        """

        def get_filter_target(row, diff_find):
            """
            :param diff_find:
            :return:
            """
            if abs(row["diff_low"]) > diff_find and row["diff_high"] > diff_find:
                return 0
            if abs(row["diff_low"]) > diff_find and row["diff_low"] < 0:
                return row["diff_low"]
            if row["diff_high"] > diff_find and row["diff_high"] > 0:
                return row["diff_high"]
            return 0

        self.df_quotes["diff_low"] = (
            self.df_quotes["Low"] / self.df_quotes["Open"]
        ) - 1
        self.df_quotes["diff_high"] = (
            self.df_quotes["High"] / self.df_quotes["Open"]
        ) - 1

        # Доп. проверка.
        self.df_quotes["diff_low"] = np.where(
            self.df_quotes["diff_low"] > 0, 0, self.df_quotes["diff_low"]
        )
        self.df_quotes["diff_high"] = np.where(
            self.df_quotes["diff_high"] < 0, 0, self.df_quotes["diff_high"]
        )

        # Создаю таргет.
        self.df_quotes["target_predict"] = self.df_quotes.apply(
            lambda row: get_filter_target(row, diff_find), axis=1
        )
        pass

    def _line_support_resistance(self):
        """
        Линии поддержки и сопротивления.
        Сейчас реализовать это довольно тяжело.
        Как минимум нужно будет делать изменения от цены открытия. И устанавливать поддержку на
        цену от цены открытия для кадого дня.
        (сначала надо будет понять что это вообще за линия)

        Так же надо учесть, что такой линии может не быть.

        :return:
        """
        pass

    def _save_final_file(self):
        """
        Сохранение итогового файла с добавленными признаками.
        :return:
        """
        name = f"{self.pair_names}_{self.count_day}d_interval-{self.interval}_diff"
        part_name = "data/processed"
        file_path = f"{part_name}/{name}.pkl"

        if not Path(part_name).exists():
            os.mkdir(part_name)

        # Сохранение данных.
        with open(file_path, "wb") as file:
            pickle.dump(self.df_quotes, file)

    def read_final_file(self):
        """
        Загрузка сохраненного файла.

        :return: Загруженные данные из файла.
        """
        name = f"{self.pair_names}_{self.count_day}d_interval-{self.interval}_diff"
        part_name = "data/processed"
        file_path = f"{part_name}/{name}.pkl"

        if Path(file_path).exists():
            # Чтение данных из файла.
            with open(file_path, "rb") as file:
                data = pickle.load(file)
            return data
        else:
            raise FileNotFoundError(f"Файл {file_path} не найден.")

    def add_date_id(self):
        self.df_quotes["Open time"] = pd.to_datetime(self.df_quotes["Open time"])
        self.df_quotes = self.df_quotes.sort_values(by="Open time", ascending=True)
        self.df_quotes = self.df_quotes.reset_index(drop=True)
        self.df_quotes = self.df_quotes.reset_index(drop=False).rename(
            columns={"index": "id_date"}
        )

    def _add_targt_class(self):
        """
        Создаю класс для прогнозирования.
        """

        def split_by_class(val):
            if val == 0:
                return 0
            elif val > 1:
                return 1
            elif val < 1:
                return -1

        self.df_quotes["target_class"] = self.df_quotes["target_predict"].apply(
            split_by_class
        )

    def prepare_data_main(self):
        self.set_work_path()
        if self.load_new_data:
            self.get_data_trading()
        else:
            self._read_save_quotes()
        self._convert_to_float()
        self._convert_to_diff_format()
        self._find_target_bar(diff_find=0.05)
        self.add_date_id()
        self._add_targt_class()
        self._save_final_file()
        self.df_quotes.to_excel("final_data.xlsx", index=False)
        return self.df_quotes


if __name__ == "__main__":
    DataPreparation(load_new_data=False).prepare_data_main()
