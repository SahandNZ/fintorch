from abc import ABC, abstractmethod

from fintorch.cross_validation.fold import Fold
from fintorch.dataset.dataset import Dataset


class CrossValidation(ABC):
    def __init__(self):
        self.__dataset: Dataset = None
        self._index: int = None

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def index(self) -> int:
        return self._index

    def set_dataset(self, dataset: Dataset):
        self.__dataset = dataset

    @abstractmethod
    def __iter__(self):
        NotImplemented()

    @abstractmethod
    def __next__(self) -> Fold:
        NotImplemented()
