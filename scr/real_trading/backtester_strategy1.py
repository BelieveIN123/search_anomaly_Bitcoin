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

matplotlib.use('Agg')  # Установка бэкенда перед импортом pyplot

# Предполагаем, что у вас есть начально обученная модель CatBoost
model = CatBoostClassifier()

class CustomPandasData(bt.feeds.PandasData):
    lines = ('month_year', 'target')
    params = (('month_year', -1),
              ('target', -1))

class CustomStrategy(bt.Strategy):
    params = (
        ('stop_loss', -0.05),
        ('take_profit', 0.05),
        ('hold_days', 3),
        ('retrain_period', 300),  # Период дообучения в днях
        ('initial_skip_days', 100),  # Количество дней для пропуска перед началом дообучения
        ('training_data_window', 100)  # Размер окна для обучения
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
        self.exit_reason = ''
        self.entry_order = None
        self.take_profit_order = None
        self.stop_loss_order = None
        self.data_history: list = []

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
            'datetime': self.data.datetime.date(0),
            'prediction': prediction[0]  # Assuming prediction is an array
        }
        self.data_history.append(prediction_info)

        price_close = self.data0.close[0]
        if self.position:
            if self.position.size > 0:
                price_buy = self.buyprice
                if len(self) >= (self.bar_executed + self.params.hold_days):
                    self.exit_reason = 'Hold days exit'
                    # Cancel existing exit orders
                    self.cancel(self.take_profit_order)
                    self.cancel(self.stop_loss_order)
                    # Place market sell order
                    self.record_exit(price_close, self.exit_reason)
                    self.order = self.sell(size=self.position.size, price=price_close)
                    return
                elif prediction == -1:
                    self.exit_reason = 'Prediction exit'
                    # Cancel existing exit orders
                    self.cancel(self.take_profit_order)
                    self.cancel(self.stop_loss_order)
                    # Place market sell order
                    self.record_exit(price_close, self.exit_reason)
                    self.order = self.sell(size=self.position.size, price=price_close)
                    return
        else:
            if prediction == 1:
                size_to_buy = self.broker.get_cash() // price_close
                price_take = price_close * (1 + self.params.take_profit)
                price_stop = price_close * (1 + self.params.stop_loss)
                self.entry_order, self.take_profit_order, self.stop_loss_order = self.buy_bracket(
                    size=size_to_buy,
                    limitprice=price_take,
                    price=price_close,
                    stopprice=price_stop
                )
                self.bar_executed = len(self)
                self.record_entry(price_close, size_to_buy)

        self.training_data.append(self.get_training_features())
        self.training_labels.append(self.get_training_label())

    def record_entry(self, price, size):
        """Запись информации о входе в позицию."""
        entry_info = {
            'datetime': self.data.datetime.date(0),
            'type': 'Buy',
            'size': size,
            'price': price,
            'value': self.broker.getvalue(),
            'trade_reason': 'Model prediction',
        }
        self.trade_history.append(entry_info)
        print(f"Entered position: {entry_info}")

    def record_exit(self, price, reason):
        """Запись информации о выходе из позиции."""
        exit_info = {
            'datetime': self.data.datetime.date(0),
            'type': 'Sell',
            'size': self.position.size,
            'price': price,
            'value': price * self.position.size + self.broker.get_cash(), # self.broker.getvalue() # Аналогично
            'trade_reason': reason,
        }
        self.trade_history.append(exit_info)
        print(f"Exited position: {exit_info}")

    def data_preparation(self, data=None):
        if data is None:
            raise ValueError('data должна иметь значения.')
        else:
            data = pd.DataFrame(data)
        data['range'] = data['high'] - data['low']
        data['change'] = data['close'] - data['open']
        return data

    def get_training_features(self):
        features = {
            'open': self.data0.open[0],
            'high': self.data0.high[0],
            'low': self.data0.low[0],
            'close': self.data0.close[0],
            'volume': self.data0.volume[0],
            'month_year':self.data0.month_year[0],
        }
        return features

    def get_training_label(self):
        target = {'target': self.data0.target[0]}
        return target

    def retrain_model(self):
        new_data = pd.DataFrame(self.training_data)
        new_labels = pd.DataFrame(self.training_labels)['target']
        prepared_data = self.data_preparation(new_data)
        model.fit(new_data, new_labels, verbose=False)

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(CustomStrategy)
    data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
    data.columns = [str.lower(col) for col in list(data)]
    data['month_year'] = data.index.day + data.index.month * 100 + data.index.year * 100 * 100
    data['target'] = data['open'] / data['open'].shift(1)
    data['target'] = data['target'].fillna(0)
    mask1 = (data['target'] > 1.02)
    data.loc[mask1, 'target'] = 1
    mask2 = (data['target'] < 0.98)
    data.loc[mask2, 'target'] = -1
    data.loc[~(mask1 | mask2), 'target'] = 0

    data_feed = CustomPandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000.0)
    cerebro.broker.setcommission(commission=0.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # Run the strategy
    strategies = cerebro.run()
    strategy = strategies[0]  # Get the instance of the strategy

    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Convert prediction history to DataFrame
    predictions_df = pd.DataFrame(strategy.data_history)
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
    path_excel = Path(r'D:\Analyst_profession\PROGECT\BTC Find Anamaly\work git\search_anomaly_Bitcoin\data history.xlsx')
    predictions_df.to_excel(path_excel)

    with matplotlib.pyplot.ioff():
        figs = cerebro.plot(iplot=False)
        for fig in figs:
            for f in fig:
                plt.figure(f.number)
                plt.savefig(f'my_strategy_plot{f.number}.png')
                plt.close(f)
