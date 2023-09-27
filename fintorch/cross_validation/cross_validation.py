from abc import ABC

from fintorch.cross_validation.fold import Fold
from fintorch.dataset.dataset import Dataset


class CrossValidation(ABC):
    def __init__(self, train_percentage: float, dev_percentage: float):
        self.__train_percentage: float = train_percentage
        self.__dev_percentage: float = dev_percentage

        self.__dataset: Dataset = None
        self._index: int = None

    @property
    def train_percentage(self) -> float:
        return self.__train_percentage

    @property
    def dev_percentage(self) -> float:
        return self.__dev_percentage

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def index(self) -> int:
        return self._index

    def set_dataset(self, dataset: Dataset):
        self.__dataset = dataset

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self) -> Fold:
        self._index += 1
        if self._index < 1:
            train_stop_index = int(len(self.dataset) * self.train_percentage)
            dev_stop_index = train_stop_index + int(len(self.dataset) * self.dev_percentage)

            # create splits
            train_set = self.dataset[:train_stop_index]
            dev_set = self.dataset[train_stop_index:dev_stop_index]
            test_set = self.dataset[dev_stop_index:]
            fold = Fold(self.index, train_set, dev_set, test_set)

            return fold

        else:
            raise StopIteration()
