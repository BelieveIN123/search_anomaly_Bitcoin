import os
from pathlib import Path
import logging as logger
from scr.real_trading.backtester_strategy1 import CustomStrategy
from scr.data.data_preparation import DataPreparation


class BacktestStrategy1():
    def __init__(self):
        pass

    def _set_work_path(self) -> Path:
        # Получаем текущий рабочий каталог
        current_directory = os.getcwd()

        # Ищем позицию, где находится "su-data-platform" в пути
        index = current_directory.rfind("search_anomaly_Bitcoin")
        index1 = current_directory.rfind("pysetup")

        # Если "su-data-platform" найден, переходим к его родительскому каталогу
        if index != -1:
            project_directory = current_directory[: index + len("search_anomaly_Bitcoin")]

            # Переходим в родительский каталог "su-data-platform"
            os.chdir(project_directory)
        elif index1 != -1:
            project_directory = current_directory[: index1 + len("pysetup")]

            # Переходим в родительский каталог "su-data-platform"
            os.chdir(project_directory)
        else:
            raise Exception("Произошла ошибка задания рабочего каталога.")

        current_directory = os.getcwd()
        print(f"Рабочая директория: {current_directory}")
        return Path(current_directory)


    def run_backtest(self):
        self._set_work_path()
        df = DataPreparation().read_final_file()
        print('df', type(df), list(df))
        # Скопировать из backtrest_strategy1 -> if __name__ == '__main__': -> # TODO
        strategy = pass


if __name__ == '__main__':
    run_test = BacktestStrategy1()
    run_test.run_backtest()

