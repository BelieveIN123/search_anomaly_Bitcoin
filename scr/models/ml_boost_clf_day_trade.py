# Примерный план по которому я двигаюсь:
from catboost import CatBoostClassifier, Pool


'''
Можно предсказывать только длинные движения.  То есть модель обучать только на изменениях выше 5%.

Проверять же на более сложном алгоритме.
К примеру
После входа ставить стол на -5%.
Также закрывать позицию через 3 дня после открытия если ничего не происходило.
Как плюс так в минус.

Это довольно базовый вариант модели.
И при реализации я затрону все важные для меня аспекты.
Ну и конечно же она может заработать.


Также критерием выхода будет за то что модель говорит на следующем баре противоположный прогноз.

'''

class MLBoostClfDayTrade:
    def __init__(self):
        self.model = CatBoostClassifier(iterations=2,
                                   depth=2,
                                   learning_rate=1,
                                   loss_function='Logloss',
                                   verbose=True)
        self.columns=[]
    def fit(self, X, y):
        self.model.fit(X[self.columns], y)

    def _check_input_data_X(self):
        pass

    def get_prediction(self, X):
        self.model.predict(X)

    def decision_trade(self):
        '''
        Условия входа в сделку.
        :return:
        '''
        pass

    def decision_close(self):
        '''
        Условия закрытия сделки.
        :return:
        '''
        pass
