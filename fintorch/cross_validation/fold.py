from typing import List

from matplotlib import pyplot as plt

from fintorch.dataset.dataset import Dataset
from fintorch.metrics import Metrics


class Fold:
    def __init__(self, index: int, train_set: Dataset, dev_set: Dataset, test_set: Dataset,
                 best_train_metrics: Metrics = None, best_dev_metrics: Metrics = None,
                 best_test_metrics: Metrics = None, best_dev_on_test_metrics: Metrics = None):
        self.__index: int = index
        self.__train_set: Dataset = train_set
        self.__dev_set: Dataset = dev_set
        self.__test_set: Dataset = test_set

        self.__best_train_metrics: Metrics = best_train_metrics
        self.__best_dev_metrics: Metrics = best_dev_metrics
        self.__best_test_metrics: Metrics = best_test_metrics
        self.__best_dev_on_test_metrics: Metrics = best_dev_on_test_metrics

        self.__train_metrics_list: List[Metrics] = []
        self.__dev_metrics_list: List[Metrics] = []
        self.__test_metrics_list: List[Metrics] = []

        self.getstate_mode: str = "experiment"

    @staticmethod
    def aggregate(folds: List):
        train_set = Dataset.aggregate([fold.train_set for fold in folds])
        dev_set = Dataset.aggregate([fold.dev_set for fold in folds])
        test_set = Dataset.aggregate([fold.test_set for fold in folds])
        best_train_metrics = Metrics.aggregate([fold.best_train_metrics for fold in folds])
        best_dev_metrics = Metrics.aggregate([fold.best_dev_metrics for fold in folds])
        best_test_metrics = Metrics.aggregate([fold.best_test_metrics for fold in folds])
        best_dev_on_test_metrics = Metrics.aggregate([fold.best_dev_on_test_metrics for fold in folds])
        fold = Fold(index=None, train_set=train_set, dev_set=dev_set, test_set=test_set,
                    best_train_metrics=best_train_metrics, best_dev_metrics=best_dev_metrics,
                    best_test_metrics=best_test_metrics, best_dev_on_test_metrics=best_dev_on_test_metrics)

        return fold

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
    def train_metrics_list(self) -> List[Metrics]:
        return self.__train_metrics_list

    @property
    def dev_metrics_list(self) -> List[Metrics]:
        return self.__dev_metrics_list

    @property
    def test_metrics_list(self) -> List[Metrics]:
        return self.__test_metrics_list

    @property
    def best_train_metrics(self) -> Metrics:
        if self.__best_train_metrics is None:
            self.__best_train_metrics = Metrics.get_best_metric(self.train_metrics_list)
        return self.__best_train_metrics

    @property
    def best_dev_metrics(self) -> Metrics:
        if self.__best_dev_metrics is None:
            self.__best_dev_metrics = Metrics.get_best_metric(self.dev_metrics_list)
        return self.__best_dev_metrics

    @property
    def best_test_metrics(self) -> Metrics:
        if self.__best_test_metrics is None:
            self.__best_test_metrics = Metrics.get_best_metric(self.test_metrics_list)
        return self.__best_test_metrics

    @property
    def best_dev_on_test_metrics(self) -> Metrics:
        if self.__best_dev_on_test_metrics is None:
            self.__best_dev_on_test_metrics = self.test_metrics_list[self.best_dev_metrics.epoch]
        return self.__best_dev_on_test_metrics

    def append_train_metrics(self, metrics: Metrics):
        self.__train_metrics_list.append(metrics)
        self.__best_train_metrics = None

    def append_dev_metrics(self, metrics: Metrics):
        self.__dev_metrics_list.append(metrics)
        self.__best_dev_metrics = None

    def append_test_metrics(self, metrics: Metrics):
        self.__test_metrics_list.append(metrics)
        self.__best_test_metrics = None

    def to(self, device: str):
        self.train_set.to(device)
        self.dev_set.to(device)
        self.test_set.to(device)

    # TODO Clean up indentations in logs
    def print_classification_logs(self):
        names = ["Train set", "Dev set", "Test set", "Best Dev on Test set"]
        sets = [self.best_train_metrics, self.best_dev_metrics, self.best_test_metrics, self.best_dev_on_test_metrics]
        for name, metrics in zip(names, sets):
            print(name)
            print("\t{:<32}{}".format("Loss", metrics.objective))
            print("\t{:<32}{}".format("Accuracy", metrics.accuracy))
            print("\t{:<32}{}\n".format("Probability Accuracy", metrics.probability_accuracy))

            print("\t{:<32}{:<16}{:<32}{:<16}{:<32}{:<16}{:<32}"
                  .format("Label \\ Measure", "Precision", "Probability Precision", "Recall", "Probability Recall",
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

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "deployment" == self.getstate_mode:
            if "_Fold__train_set" in dct:
                del dct["_Fold__train_set"]
            if "_Fold__dev_set" in dct:
                del dct["_Fold__dev_set"]

            if "_Fold__train_metrics_list" in dct:
                del dct["_Fold__train_metrics_list"]
            if "_Fold__dev_metrics_list" in dct:
                del dct["_Fold__dev_metrics_list"]
            if "_Fold__test_metrics_list" in dct:
                del dct["_Fold__test_metrics_list"]

            if "_Fold__best_train_metrics" in dct:
                del dct["_Fold__best_train_metrics"]
            if "_Fold__best_dev_metrics" in dct:
                del dct["_Fold__best_dev_metrics"]
            if "_Fold__best_test_metrics" in dct:
                del dct["_Fold__best_test_metrics"]

        return dct
