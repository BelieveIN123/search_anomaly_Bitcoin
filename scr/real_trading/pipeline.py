'''
Я работаю с python, библиотеки backtrader, catboost

Я хочу предсказывать только длинные движения.  То есть модель обучать только на изменениях выше 5%.
У меня есть скрипт который добвляет итрересующий таргет и другие колонки.
Таргет представляет собой 3 класса.

Модель:
catboost classifier

Условие входа:
- Если модель прогнозирует класс_0 или класс-2

Условия выхода:
- Если позиция открыта 3 дня.
- Если позиция открыта, но прогноз в другую сторону на следующем дне.
- Если я позиция открыта в плюс 5%
- Если позиция открыта в минус 5% (стоп лосс)
'''


import backtrader as bt
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import datetime

# Предполагаем, что у вас есть обученная модель CatBoost
model = CatBoostClassifier()
model.load_model('path_to_your_model.cbm')

class CustomStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.05),
        ('take_profit', 0.05),
        ('hold_days', 3),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.hold_day_count = 0

    def next(self):
        # Получаем данные для прогноза
        data = self.get_data_for_prediction()
        prediction = model.predict(data)

        if self.order:
            # Условия выхода
            if len(self) >= (self.bar_executed + self.params.hold_days):
                self.order = self.sell()
                return

            current_position = self.dataclose[0] / self.buyprice - 1
            if current_position >= self.params.take_profit:
                self.order = self.sell()
                return
            elif current_position <= -self.params.stop_loss:
                self.order = self.sell()
                return

            if prediction == 1:  # Прогноз в другую сторону
                self.order = self.sell()
                return
        else:
            # Условия входа
            if prediction == 0 or prediction == 2:
                self.order = self.buy()
                self.buyprice = self.dataclose[0]
                self.bar_executed = len(self)
                self.hold_day_count = 0

    def get_data_for_prediction(self):
        # Здесь реализуйте сбор данных для прогноза, например, используйте pandas DataFrame
        # data = pd.DataFrame(...)

        # Пример использования последних значений для прогноза
        data = pd.DataFrame({
            'feature1': [self.dataclose[0]],  # Пример использования текущей цены
            # Добавьте здесь другие необходимые признаки
        })

        return data

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(CustomStrategy)

    # Загрузка данных
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime.datetime(2022, 1, 1),
        todate=datetime.datetime(2023, 1, 1)
    )

    cerebro.adddata(data)

    # Начальные деньги
    cerebro.broker.set_cash(10000.0)

    # Печать начальных условий
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Запуск backtrader
    cerebro.run()

    # Печать конечного результата
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
