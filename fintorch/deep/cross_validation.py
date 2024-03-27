import math
from datetime import datetime
from typing import Iterator, Union

from fintorch.deep.dtype import Fold
from fintorch.enum import TimeFrame
from fintorch.utils.hash import static_list_hash
from fintorch.utils.timestamp import round_timestamp


class CrossValidation:
    def __init__(
            self,
            interval: TimeFrame,
            train_length: int,
            val_length: int,
            test_length: int
    ):
        self.__interval: TimeFrame = interval
        self.__train_length: int = train_length
        self.__val_length: int = val_length
        self.__test_length: int = test_length

        self.__first_timestamp: int = -1
        self.__last_timestamp: int = -1
        self.__start_timestamp: int = -1
        self.__stop_timestamp: int = -1
        self.__samples_count: int = -1
        self.__folds_count = -1
        self.__index: int = -1

        self.__static_hash: int = static_list_hash(
            [
                int(self.interval),
                self.train_length,
                self.val_length,
                self.test_length
            ]
        )

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
    def first_timestamp(self) -> int:
        return self.__first_timestamp

    @property
    def last_timestamp(self) -> int:
        return self.__last_timestamp

    @property
    def start_timestamp(self) -> int:
        return self.__start_timestamp

    @property
    def stop_timestamp(self) -> int:
        return self.__stop_timestamp

    @property
    def first_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.first_timestamp)

    @property
    def last_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.last_timestamp)

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.stop_timestamp)

    @property
    def samples_count(self) -> int:
        return self.__samples_count

    @property
    def folds_count(self) -> int:
        return self.__folds_count

    @property
    def index(self) -> int:
        return self.__index

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def __call__(
            self,
            first_timestamp: int,
            last_timestamp: int,
            start_timestamp: Union[int, None] = None,
            stop_timestamp: Union[int, None] = None
    ) -> Iterator:
        # replace none values
        start_timestamp = start_timestamp if start_timestamp is not None else 0
        stop_timestamp = stop_timestamp if stop_timestamp is not None else int(datetime.now().timestamp())

        # round timestamps
        first_timestamp = round_timestamp(timestamp=first_timestamp, time_frame=self.interval)
        last_timestamp = round_timestamp(timestamp=last_timestamp, time_frame=self.interval)
        start_timestamp = round_timestamp(timestamp=start_timestamp, time_frame=self.interval)
        stop_timestamp = round_timestamp(timestamp=stop_timestamp, time_frame=self.interval)

        self.__first_timestamp = int(first_timestamp + (self.train_length + self.val_length) * self.interval)
        self.__last_timestamp = last_timestamp
        self.__start_timestamp = max(first_timestamp, start_timestamp)
        self.__stop_timestamp = min(last_timestamp, stop_timestamp)

        return self.__iter__()

    def __iter__(self):
        self.__samples_count = (self.stop_timestamp - self.start_timestamp) // int(self.interval)
        self.__folds_count = math.ceil(self.samples_count / self.test_length)
        self.__index = -1

        return self

    def __next__(self) -> Fold:
        self.__index += 1
        if self.index < self.folds_count:
            test_start_timestamp = int(self.start_timestamp + self.index * self.test_length * self.interval)
            test_stop_timestamp = int(test_start_timestamp + self.test_length * self.interval)
            test_timestamps = list(range(test_start_timestamp, test_stop_timestamp, int(self.interval)))

            val_stop_timestamp = test_start_timestamp
            val_start_timestamp = int(val_stop_timestamp - self.val_length * self.interval)
            val_timestamps = list(range(val_start_timestamp, val_stop_timestamp, int(self.interval)))

            train_stop_timestamp = val_start_timestamp
            train_start_timestamp = int(train_stop_timestamp - self.train_length * self.interval)
            train_timestamps = list(range(train_start_timestamp, train_stop_timestamp, int(self.interval)))

            fold = Fold(
                index=self.index,
                train_timestamps=train_timestamps,
                val_timestamps=val_timestamps,
                test_timestamps=test_timestamps
            )

            return fold
        else:
            raise StopIteration
