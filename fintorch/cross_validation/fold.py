from typing import List

from matplotlib import pyplot as plt

from fintorch.dataset.dataset import Dataset
from fintorch.metrics import Metrics


class Fold:
    def __init__(self, index: int, train_set: Dataset, dev_set: Dataset, test_set: Dataset):
        self.__index: int = index
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
    def index(self) -> int:
        return self.__index

    @property
    def train_set(self) -> Dataset:
        return self.__train_set

    @property
    def dev_set(self) -> Dataset:
        return self.__dev_set

    @property
    def test_set(self) -> Dataset:
        return self.__test_set

    @property
    def best_dev_on_test_metrics(self) -> Metrics:
        return self.test_metrics_list[self.best_dev_metrics.epoch]

    def print_classification_logs(self):
        names = ["Train set", "Dev set", "Test set", "Best Dev Model on Test set"]
        sets = [self.best_train_metrics, self.best_dev_metrics, self.best_test_metrics, self.best_dev_on_test_metrics]
        for name, metrics in zip(names, sets):
            print(name)
            print("\t{:<32}{}".format("Loss", metrics.objective))
            print("\t{:<32}{}".format("Accuracy", metrics.accuracy))
            print("\t{:<32}{}\n".format("Probability Accuracy", metrics.probability_accuracy))

            print("\t{:<32}{:<16}{:<32}{:<16}{:<32}{:<16}{:<32}".
                  format("Label \\ Measure", "Precision", "Probability Precision", "Recall", "Probability Recall",
                         "F1-score", "Probability F1-score"))
            for label in range(self.dev_set.label_transform.num_classes):
                p = metrics.precision(label=label)
                pp = metrics.f1(label=label)
                r = metrics.recall(label=label)
                pr = metrics.probability_recall(label=label)
                f1 = metrics.f1(label=label)
                pf1 = metrics.probability_f1(label=label)
                print("\t{:<32}{:<16}{:<32}{:<16}{:<32}{:<16}{:<32}".format(label, p, pp, r, pr, f1, pf1))
            print()

    def show_learning_curve_plot(self):
        plt.figure(figsize=(10, 5))

        plt.plot([m.objective for m in self.train_metrics_list], 'b', label='train objective')
        plt.plot([m.objective for m in self.dev_metrics_list], 'y', label='dev objective')
        plt.plot([m.objective for m in self.test_metrics_list], 'r', label='test objective')

        plt.legend()
        plt.grid()
        plt.show()
