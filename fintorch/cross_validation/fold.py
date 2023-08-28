from typing import List

from matplotlib import pyplot as plt

from fintorch.dataset.dataset import Dataset
from fintorch.metrics import Metrics


class Fold:
    def __init__(self, train_set: Dataset, dev_set: Dataset, test_set: Dataset):
        self.__train_set: Dataset = train_set
        self.__dev_set: Dataset = dev_set
        self.__test_set: Dataset = test_set

        self.train_metrics_list: List[Metrics] = []
        self.dev_metrics_list: List[Metrics] = []
        self.test_metrics_list: List[Metrics] = []

        self.best_train_metrics: Metrics = None
        self.best_dev_metrics: Metrics = None
        self.best_test_metrics: Metrics = None

    @property
    def train_set(self) -> Dataset:
        return self.__train_set

    @property
    def dev_set(self) -> Dataset:
        return self.__dev_set

    @property
    def test_set(self) -> Dataset:
        return self.__test_set

    def show_plot(self):
        fig = plt.figure(figsize=(10, 5))

        plt.plot([m.objective for m in self.train_metrics_list], 'b', label='train obejctive')
        plt.plot([m.objective for m in self.dev_metrics_list], 'y', label='dev obejctive')
        plt.plot([m.objective for m in self.test_metrics_list], 'r', label='test obejctive')

        plt.legend()
        plt.grid()
        plt.show()
