from abc import ABC
from datetime import datetime
from typing import Iterator

from ..dtype import Dataset, Fold
from ...enum import TimeFrame


class CrossValidation(ABC):
    def __init__(self, train_percentage: float, dev_percentage: float):
        self.__train_percentage: float = train_percentage
        self.__dev_percentage: float = dev_percentage

        self.__dataset: Dataset = None

        self._train_length: int = None
        self._dev_length: int = None
        self._test_length: int = None
        self._folds_count: int = None
        self._index: int = -1

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
    def time_frame(self) -> TimeFrame:
        return self.dataset.sampling_time_frame

    @property
    def start_date(self) -> datetime:
        return self.dataset.start_date

    @property
    def stop_date(self) -> datetime:
        return self.dataset.stop_date

    @property
    def samples_count(self) -> int:
        return len(self.dataset)

    @property
    def train_length(self) -> int:
        return self._train_length

    @property
    def dev_length(self) -> int:
        return self._dev_length

    @property
    def test_length(self) -> int:
        return self._test_length

    @property
    def folds_count(self) -> int:
        return self._folds_count

    @property
    def index(self) -> int:
        return self._index

    def _create_fold_by_start_timestamp(self, start_timestamp: int) -> Fold:
        train_stop_timestamp = int(start_timestamp + self.train_length * self.time_frame)
        dev_stop_timestamp = int(train_stop_timestamp + self.dev_length * self.time_frame)
        test_stop_timestamp = int(dev_stop_timestamp + self.test_length * self.time_frame)

        train_timestamps = list(range(start_timestamp, train_stop_timestamp, self.time_frame))
        dev_timestamps = list(range(train_stop_timestamp, dev_stop_timestamp, self.time_frame))
        test_timestamps = list(range(dev_stop_timestamp, test_stop_timestamp, self.time_frame))

        fold = Fold(train_timestamps, dev_timestamps, test_timestamps)
        return fold

    def __call__(self, dataset: Dataset) -> Iterator:
        self.__dataset = dataset

        self._train_length = int(self.samples_count * self.train_percentage // 100)
        self._dev_length = int(self.samples_count * self.dev_percentage // 100)
        self._test_length = int(self.samples_count - self.train_length - self.dev_length)

        return self.__iter__()

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self) -> Fold:
        self._index += 1
        if self.index < 1:
            start_timestamp = int(self.start_date.timestamp())
            fold = self._create_fold_by_start_timestamp(start_timestamp=start_timestamp)
            return fold
        else:
            raise StopIteration
