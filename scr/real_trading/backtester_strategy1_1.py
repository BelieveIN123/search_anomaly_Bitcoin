import backtrader as bt
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import datetime
from collections import deque
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")  # Установка бэкенда перед импортом pyplot

# Предполагаем, что у вас есть начально обученная модель CatBoost
model = CatBoostClassifier()


# class CustomPandasData(bt.feeds.PandasData):
#     lines = ("month_year", "target")
#     params = (("month_year", -1), ("target", -1))

columns_main_backtr = ["date", "open", "high", "low", "close", "volume"]

columns_for_backtrader = [
    "date",
    "open",
    "close",
    "open_diff",
    "high_diff",
    "low_diff",
    "close_diff",
    "volume_diff",
    "quote_asset_volume_diff",
    "number_of_trades_diff",
    # "diff_low",
    # "diff_high",
    "id_day",
    "target_predict",
    "target_class",
]


def create_custom_pandas_data(custom_lines):
    # Build a dictionary of parameters, mapping each line to the column name
    set_param = ["date"]
    params_dict = {k: k for k in custom_lines if k not in set_param}

    class CustomPandasData1(bt.feeds.PandasData):
        lines = tuple(custom_lines)
        params = (("date", None),) + tuple(params_dict.items())
        print("params", params)

    return CustomPandasData1

    # Устанавливаем индексы для обязательных параметров


class CustomStrategy(bt.Strategy):
    params = (
        ("stop_loss", -0.05),
        ("take_profit", 0.05),
        ("hold_days", 3),
        ("retrain_period", 100),  # Период дообучения в днях
        (
            "initial_skip_days",
            100,
        ),  # Количество дней для пропуска перед началом дообучения
        ("training_data_window", 100),  # Размер окна для обучения
        ("columns_for_fit_model", None),
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.order = None
        self.buyprice = None
        self.bar_executed = None
        self.last_retrain = -(self.params.retrain_period)
        self.trade_history = []  # История сделок
        self.training_data = deque(maxlen=self.params.training_data_window)
        self.training_labels = deque(maxlen=self.params.training_data_window)
        self.exit_reason = ""
        self.entry_order = None
        self.take_profit_order = None
        self.stop_loss_order = None
        self.data_history: list = []
        self.columns_for_fit_model = self.params.columns_for_fit_model
        #     [
        #     "id_day",
        #     "open_diff",
        #     "high_diff",
        #     "low_diff",
        #     "close_diff",
        #     "volume_diff",
        #     "quote_asset_volume_diff",
        #     "number_of_trades_diff",
        # ]
        self.column_fit_target = "target_class"

    def next(self):
        if len(self) < self.params.initial_skip_days:
            self.training_data.append(self.get_training_features())
            self.training_labels.append(self.get_training_label())
            return

        if (len(self) - self.last_retrain) >= self.params.retrain_period:
            self.retrain_model()
            self.last_retrain = len(self)

        iter_predict_data = pd.DataFrame([self.get_training_features()])
        data = self.data_preparation(iter_predict_data)
        prediction = model.predict(data)

        # Record the date and prediction
        prediction_info = {
            "datetime": self.data.datetime.date(0),
            "prediction": prediction[0],  # Assuming prediction is an array
        }
        self.data_history.append(prediction_info)

        price_close = self.data0.close[0]
        if self.position:
            if self.position.size > 0:  # Длинная позиция
                price_buy = self.buyprice
                if len(self) >= (self.bar_executed + self.params.hold_days):
                    self.exit_reason = "Hold days exit (Long)"
                    self.close_position(price_close)
                elif prediction == -1:  # Сигнал на продажу
                    self.exit_reason = "Prediction exit (Short signal)"
                    self.close_position(price_close)
            elif self.position.size < 0:  # Короткая позиция
                # price_sell = self.sellprice
                if len(self) >= (self.bar_executed + self.params.hold_days):
                    self.exit_reason = "Hold days exit (Short)"
                    self.close_position(price_close)
                elif prediction == 1:  # Сигнал на покупку
                    self.exit_reason = "Prediction exit (Long signal)"
                    self.close_position(price_close)
        else:
            if prediction == 1:  # Открытие длинной позиции
                size_to_buy = self.broker.get_cash() // price_close
                price_take = price_close * (1 + self.params.take_profit)
                price_stop = price_close * (1 + self.params.stop_loss)
                self.entry_order, self.take_profit_order, self.stop_loss_order = (
                    self.buy_bracket(
                        size=size_to_buy,
                        limitprice=price_take,
                        price=price_close,
                        stopprice=price_stop,
                    )
                )
                self.bar_executed = len(self)
                self.record_entry(price_close, size_to_buy, "Buy")
            elif prediction == -1:  # Открытие короткой позиции
                size_to_sell = self.broker.get_cash() // price_close
                price_take = price_close * (1 - self.params.take_profit)
                price_stop = price_close * (1 - self.params.stop_loss)
                self.entry_order, self.take_profit_order, self.stop_loss_order = (
                    self.sell_bracket(
                        size=size_to_sell,
                        limitprice=price_take,
                        price=price_close,
                        stopprice=price_stop,
                    )
                )
                self.bar_executed = len(self)
                self.record_entry(price_close, size_to_sell, "Sell")

        self.training_data.append(self.get_training_features())
        self.training_labels.append(self.get_training_label())

    def close_position(self, price_close):
        """Закрытие текущей позиции с записью информации."""
        self.cancel(self.take_profit_order)
        self.cancel(self.stop_loss_order)
        self.record_exit(price_close, self.exit_reason)
        if self.position.size > 0:
            self.order = self.sell(size=self.position.size, price=price_close)
        elif self.position.size < 0:
            self.order = self.buy(size=-self.position.size, price=price_close)

    def record_entry(self, price, size, position_type):
        """Запись информации о входе в позицию."""
        entry_info = {
            "datetime": self.data.datetime.date(0),
            "type": position_type,
            "size": size,
            "price": price,
            "value": self.broker.getvalue(),
            "trade_reason": "Model prediction",
        }
        self.trade_history.append(entry_info)
        print(f"Entered position: {entry_info}")

    def record_exit(self, price, reason):
        """Запись информации о выходе из позиции."""
        exit_info = {
            "datetime": self.data.datetime.date(0),
            "type": "Sell",
            "size": self.position.size,
            "price": price,
            "value": price * self.position.size
            + self.broker.get_cash(),  # self.broker.getvalue() # Аналогично
            "trade_reason": reason,
        }
        self.trade_history.append(exit_info)
        print(f"Exited position: {exit_info}")

    def data_preparation(self, data=None):
        if data is None:
            raise ValueError("data должна иметь значения.")
        else:
            data = pd.DataFrame(data)
        # data["range"] = data["high"] - data["low"]
        # data["change"] = data["close"] - data["open"]
        return data

    def get_training_features(self):
        features = {
            col: getattr(self.data0, col)[0] for col in self.columns_for_fit_model
        }
        return features

    def get_training_label(self):
        target = {"target": self.data0.target_class[0]}
        return target

    def retrain_model(self):
        new_data = pd.DataFrame(self.training_data)
        new_labels = pd.DataFrame(self.training_labels)["target"]
        prepared_data = self.data_preparation(new_data)
        model.fit(new_data, new_labels, verbose=False)


def run_backtest_and_save_plot(
    strategy_class, data, all_column_in_strategy, column_for_fit
):
    # all_column_in_strategy = columns_for_backtrader + columns_for_fit

    data = data.set_index("date")
    custompandasdata = create_custom_pandas_data(custom_lines=all_column_in_strategy)
    data_feed = custompandasdata(dataname=data)
    print("Создана модель данных")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, columns_for_fit_model=column_for_fit)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(1000000.0)
    cerebro.broker.setcommission(commission=0.002)
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Запуск стратегии
    strategies = cerebro.run()
    strategy = strategies[0]  # Получение экземпляра стратегии

    print("Ending Portfolio Value: %.2f" % cerebro.broker.getvalue())

    # Сохранение истории предсказаний в Excel
    predictions_df = pd.DataFrame(strategy.data_history)
    path_excel = Path(
        r"D:\Analyst_profession\PROGECT\BTC Find Anamaly\work git\search_anomaly_Bitcoin\data history.xlsx"
    )
    predictions_df.to_excel(path_excel)

    # Сохранение графиков
    with matplotlib.pyplot.ioff():
        figs = cerebro.plot(iplot=False)
        for fig in figs:
            for f in fig:
                plt.figure(f.number)
                plt.savefig(f"my_strategy_plot{f.number}.png")
                plt.close(f)
