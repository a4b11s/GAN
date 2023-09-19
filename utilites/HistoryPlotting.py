import csv
from typing import Literal

import matplotlib.pyplot as plt


class HistoryPlotting:
    def __init__(self, file_path: str, file_type: Literal["csv"] = "csv"):
        self.file_path = file_path
        self.file_type = file_type
        self.history = None

    @staticmethod
    def _load_csv(file_path):
        with open(file_path, mode="r", newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

            data = {}
            for header in headers:
                if header == "epoch":
                    continue
                data[header] = []

            for row in reader:
                for i, value in enumerate(row):
                    if i == 0:
                        continue
                    data[headers[i]].append(value)

        return data

    def __load_data(self):
        if self.file_type == "csv":
            self.history = self._load_csv(self.file_path)

    def plot(self, is_save: bool = False, save_dir: str = "./", plot_type: Literal["plot"] = "plot"):
        if self.history is None:
            self.__load_data()
        if plot_type == "plot":
            for metriks in self.history.keys():
                plt.plot(self.history[metriks], label=metriks)
            plt.legend()

        if is_save:
            plt.savefig(save_dir)
        else:
            plt.show()
        plt.close()
