import os
from pathlib import Path
import logging as logger
from scr.real_trading.backtester_strategy1_1 import CustomStrategy
from scr.data.data_preparation1_1 import DataPreparation
from scr.real_trading.backtester_strategy1_1 import (
    run_backtest_and_save_plot,
    CustomStrategy,
)


class BacktestStrategy1:
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
            project_directory = current_directory[
                : index + len("search_anomaly_Bitcoin")
            ]

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
        df, add_columns = DataPreparation().read_final_file()
        print("df", type(df), list(df))
        df[
            [
                "date",
                "id_day",
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
            + add_columns
        ]

        all_column_in_strategy = list(df)
        strategy_result = run_backtest_and_save_plot(
            CustomStrategy, df, all_column_in_strategy
        )


if __name__ == "__main__":
    run_test = BacktestStrategy1()
    run_test.run_backtest()
