import backtrader as bt
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import datetime
from collections import deque
import yfinance as yf

# Предполагаем, что у вас есть начально обученная модель CatBoost
model = CatBoostClassifier()
# model.load_model('path_to_your_model.cbm')  # Изначально модель не загружается

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
        # self.dataclose = self.datas[0].close
        # self.dataopen = self.datas[0].open
        # self.datahigh = self.datas[0].high
        # self.datalow = self.datas[0].low
        # self.datavolume = self.datas[0].volume
        self.order = None
        self.buyprice = None
        self.bar_executed = None
        self.last_retrain = -(self.params.retrain_period)
        # В место 0. Чтобы на первом периоде происходило обучение.

        # Используем очередь с фиксированным размером для хранения новых данных
        self.training_data = deque(maxlen=self.params.training_data_window)
        self.training_labels = deque(maxlen=self.params.training_data_window)

    def next(self):
        # Пропуск первых n дней для начального накопления данных
        if len(self) < self.params.initial_skip_days:
            # Сохранение данных для последующего обучения
            self.training_data.append(self.get_training_features())
            self.training_labels.append(self.get_training_label())
            return

        # Ежедневное или еженедельное дообучение
        if (len(self) - self.last_retrain) >= self.params.retrain_period:
            self.retrain_model()
            self.last_retrain = len(self)

        # Получаем данные для прогноза (один временной период)
        iter_predict_data = pd.DataFrame([self.get_training_features()])
        data = self.data_preparation(iter_predict_data)  # TODO поправить. Выходные данные Nan!!!
        prediction = model.predict(data)

        # if self.order:  # TODO - Не правильное выставление ордеров.
        price_close = self.data0.close[0]
        if self.position:
            if self.position.size > 0:  # Закрытие длинной позиции
                price_buy = self.buyprice   # Цена входа в позицию.

                # Условия выхода
                ## Выход если hold_days дней в позиции.
                if len(self) >= (self.bar_executed + self.params.hold_days):
                    self.order = self.sell(size=self.position.size, price=price_close)
                    return
                ## Выход если прогоз в противо фазую.
                elif prediction == -1:
                    self.order = self.sell(size=self.position.size, price=price_close)
                    return
                # elif max_diff_position_top >= self.params.take_profit:
                #     pass
                # elif max_diff_position_bot <= self.params.stop_loss:  # TODO
                #     pass

            # elif self.position.size < 0:  # Закрытие короткой позиции # TODO
            #     # Условия выхода # 1
            #     if len(self) >= (self.bar_executed + self.params.hold_days):
            #         self.order = self.buy(size=abs(self.position.size), price=self.dataclose[0])

            # # Условия выхода # 1
            # if len(self) >= (self.bar_executed + self.params.hold_days):
            #     self.order = self.sell()
            #     return

            # current_position = self.dataclose[0] / self.buyprice - 1
            # if current_position >= self.params.take_profit:
            #     self.order = self.sell()
            #     return
            # elif current_position <= -self.params.stop_loss:
            #     self.order = self.sell()
            #     return
            #
            # if prediction == 1:  # Прогноз в другую сторону
            #     self.order = self.sell()
            #     return
        else:
            # Условия входа
            if prediction == 1:
                price_take = price_close * (1 + self.params.take_profit)
                price_stop = price_close * (1 + self.params.stop_loss)
                self.order = self.buy_bracket(limitprice=price_take, price=price_close, stopprice=price_stop)
                self.bar_executed = len(self)

        # Сохранение данных для последующего обучения
        self.training_data.append(self.get_training_features())
        self.training_labels.append(self.get_training_label())

    def data_preparation(self, data=None):
        # Собираем и нормируем данные
        if data is None:
            raise ValueError('data должна иметь значения.')
        else:
            data = pd.DataFrame(data)

        # # Нормализация данных
        # data = (data - data.mean()) / data.std()

        # Добавление новых параметров (пример: разница между high и low)
        data['range'] = data['high'] - data['low']
        data['change'] = data['close'] - data['open']

        return data

    def get_training_features(self):
        # Собираем и нормируем данные для обучения
        # features = {
        #     'open': self.dataopen[0],
        #     # 'high': self.datahigh[0],
        #     # 'low': self.datalow[0],
        #     # 'close': self.dataclose[0],
        #     # 'volume': self.datavolume[0]
        # }
        features = {
            'open': self.data0.open[0],
            'high': self.data0.high[0],
            'low': self.data0.low[0],
            'close': self.data0.close[0],
            'volume': self.data0.volume[0],
            'month_year':self.data0.month_year[0],
        }

        # # Нормализация данных
        # features = {key: (value - pd.Series(value).mean()) / pd.Series(value).std() for key, value in features.items()}
        #
        # # Добавление новых параметров (пример: разница между high и low)
        # features['range'] = features['high'] - features['low']
        # features['change'] = features['close'] - features['open']

        return features

    def get_training_label(self):
        # Здесь реализуйте логику получения метки для обучения
        # Например, это может быть следующий день или целевой класс
        target = {
            'target': self.data0.target[0],
        }
        return target

    def retrain_model(self):
        # Дообучение модели
        new_data = pd.DataFrame(self.training_data)
        new_labels = pd.DataFrame(self.training_labels)['target']

        # Преобразование данных перед fit
        prepared_data = self.data_preparation(new_data)

        # Дообучение модели
        # model.fit(new_data, new_labels, verbose=False, use_best_model=False, init_model=model)
        model.fit(new_data, new_labels, verbose=False)

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(CustomStrategy)

    # Загрузка данных через yfinance
    data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')
    data.columns = [str.lower(col) for col in list(data)]
    print('data', list(data))
    # data = (data.sort_values(['date'])
    #         .reset_index(drop=True))
    data['month_year'] = data.index.day + data.index.month * 100 + data.index.year * 100 * 100
    data['target'] = data['open'] / data['open'].shift(1)
    data['target'] = data['target'].fillna(0)
    mask1 = (data['target'] > 1.02)
    data.loc[mask1, 'target'] = 1
    mask2 = (data['target'] < 0.98)
    data.loc[mask2, 'target'] = -1
    print((~(mask1 | mask2)).sum())
    data.loc[~(mask1 | mask2), 'target'] = 0

    # data = data.reset_index(drop=False)
    # data['date'] = pd.to_datetime(data['date'])

    data_feed = CustomPandasData(dataname=data)

    cerebro.adddata(data_feed)

    # Начальные деньги
    cerebro.broker.set_cash(10000.0)

    # Печать начальных условий
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Запуск backtrader
    cerebro.run()

    # Печать конечного результата
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
