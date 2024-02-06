import math
from abc import ABC
from datetime import datetime
from typing import Iterator, List

from fintorch.deep.dtype import Dataset, Fold
from fintorch.enum import TimeFrame


class CrossValidation(ABC):
    def __init__(self, dataset: Dataset, train_length: int = 25920, val_length: int = 8640, test_length: int = 17280):
        self.__dataset: Dataset = dataset
        self.__train_length: int = train_length
        self.__val_length: int = val_length
        self.__test_length: int = test_length

        self.__test_timestamps: List[int] = None
        self.__folds_count: int = None
        self.__index: int = None

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def interval(self) -> TimeFrame:
        return self.dataset.interval

    @property
    def train_length(self) -> int:
        return self.__train_length

    @property
    def val_length(self) -> int:
        return self.__val_length

    @property
    def test_length(self) -> int:
        return self.__test_length

    @property
    def folds_count(self) -> int:
        return self.__folds_count

    @property
    def index(self) -> int:
        return self.__index

    def __call__(self, test_timestamps: List[int]) -> Iterator:
        self.__test_timestamps = test_timestamps
        return self.__iter__()

    def __iter__(self):
        self.__folds_count = math.ceil(len(self.__test_timestamps) / self.__test_length)
        self.__index = -1
        return self

    def __next__(self) -> Fold:
        self.__index += 1
        if self.index < self.folds_count:
            test_start_index = self.index * self.test_length
            test_stop_index = test_start_index + self.test_length
            test_timestamps = self.__test_timestamps[test_start_index: test_stop_index]

            val_stop_timestamp = test_timestamps[0]
            val_start_timestamp = val_stop_timestamp - self.val_length * self.interval
            val_timestamps = list(range(val_start_timestamp, test_timestamps[0], self.interval))

            train_stop_timestamp = val_start_timestamp
            train_start_timestamp = train_stop_timestamp - self.train_length * self.interval
            train_timestamps = list(range(train_start_timestamp, train_stop_timestamp, self.interval))

            fold = Fold(index=self.index, train_timestamps=train_timestamps, val_timestamps=val_timestamps,
                        test_timestamps=test_timestamps)
            return fold

        else:
            raise StopIteration
