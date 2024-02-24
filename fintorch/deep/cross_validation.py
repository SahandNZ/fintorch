import math
from datetime import datetime, timedelta
from typing import Iterator

from fintorch.deep.dtype import Fold
from fintorch.enum import TimeFrame
from fintorch.utils.hash import static_list_hash


class CrossValidation:
    def __init__(
            self,
            interval: TimeFrame,
            train_length: int = 25920,
            val_length: int = 8640,
            test_length: int = 17280
    ):
        self.__interval: TimeFrame = interval
        self.__train_length: int = train_length
        self.__val_length: int = val_length
        self.__test_length: int = test_length

        self.__test_start_timestamp: int = -1
        self.__folds_count = -1
        self.__index: int = -1

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

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
    def fold_length(self) -> int:
        return self.train_length + self.val_length + self.test_length

    @property
    def test_start_timestamp(self) -> int:
        return self.__test_start_timestamp

    @property
    def test_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.__test_start_timestamp)

    @property
    def folds_count(self) -> int:
        return self.__folds_count

    @property
    def index(self) -> int:
        return self.__index

    def __call__(self, first_valid_timestamp: int) -> Iterator:
        first_valid_timestamp = first_valid_timestamp + (self.train_length + self.val_length) * self.interval
        self.__test_start_timestamp = int(first_valid_timestamp // int(self.interval) * int(self.interval))

        return self.__iter__()

    def __iter__(self):
        current_open_timestamp = datetime.now().timestamp() // int(self.interval) * int(self.interval)
        samples_count = (current_open_timestamp - self.test_start_timestamp) // int(self.interval)
        folds_count = math.ceil(samples_count / self.test_length)

        self.__folds_count = folds_count
        self.__index = -1

        return self

    def __next__(self) -> Fold:
        self.__index += 1
        if self.index < self.folds_count:
            test_start_timestamp = int(self.test_start_timestamp + self.index * self.test_length * self.interval)
            test_stop_timestamp = int(test_start_timestamp + self.test_length * self.interval)
            test_timestamps = list(range(test_start_timestamp, test_stop_timestamp, int(self.interval)))

            val_stop_timestamp = test_start_timestamp
            val_start_timestamp = int(val_stop_timestamp - self.val_length * self.interval)
            val_timestamps = list(range(val_start_timestamp, val_stop_timestamp, int(self.interval)))

            train_stop_timestamp = val_start_timestamp
            train_start_timestamp = int(train_stop_timestamp - self.train_length * self.interval)
            train_timestamps = list(range(train_start_timestamp, train_stop_timestamp, int(self.interval)))

            fold = Fold(
                train_timestamps=train_timestamps,
                val_timestamps=val_timestamps,
                test_timestamps=test_timestamps
            )

            return fold
        else:
            raise StopIteration

    def __hash__(self):
        return static_list_hash(
            [
                int(self.interval),
                self.train_length,
                self.val_length,
                self.test_length
            ]
        )
