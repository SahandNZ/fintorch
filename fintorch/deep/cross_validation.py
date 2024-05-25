import math
from datetime import datetime, timedelta
from typing import Iterator, Union

from .status import Fold
from ..enum import TimeFrame
from ..utils.hash import static_list_hash
from ..utils.timestamp import floor_timestamp, ceil_timestamp


class CrossValidation:
    def __init__(
            self,
            time_frame: TimeFrame,
            train_period: TimeFrame,
            val_period: TimeFrame,
            test_period: TimeFrame
    ):
        self.__time_frame: TimeFrame = time_frame
        self.__train_period: TimeFrame = train_period
        self.__val_period: TimeFrame = val_period
        self.__test_period: TimeFrame = test_period

        self.__train_length: int = int(train_period / float(time_frame))
        self.__val_length: int = int(val_period / float(time_frame))
        self.__test_length: int = int(test_period / float(time_frame))

        self.__first_timestamp: float = -1
        self.__last_timestamp: float = -1
        self.__start_timestamp: int = -1
        self.__stop_timestamp: float = -1
        self.__samples_count: int = -1
        self.__folds_count = -1
        self.__index: int = -1

        self.__static_hash: int = static_list_hash(
            [
                int(self.time_frame),
                int(self.train_period),
                int(self.val_period),
                int(self.test_period)
            ]
        )

    @property
    def time_frame(self) -> TimeFrame:
        return self.__time_frame

    @property
    def train_period(self) -> TimeFrame:
        return self.__train_period

    @property
    def val_period(self) -> TimeFrame:
        return self.__val_period

    @property
    def test_period(self) -> TimeFrame:
        return self.__test_period

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
    def first_timestamp(self) -> float:
        return self.__first_timestamp

    @property
    def last_timestamp(self) -> float:
        return self.__last_timestamp

    @property
    def start_timestamp(self) -> float:
        return self.__start_timestamp

    @property
    def stop_timestamp(self) -> float:
        return self.__stop_timestamp

    @property
    def first_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.first_timestamp / 1000)

    @property
    def last_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.last_timestamp / 1000)

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp / 1000)

    @property
    def stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.stop_timestamp / 1000)

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
        stop_timestamp = stop_timestamp if stop_timestamp is not None else datetime.now().timestamp()

        # round timestamps
        first_timestamp = floor_timestamp(timestamp=first_timestamp, time_frame=self.time_frame)
        last_timestamp = floor_timestamp(timestamp=last_timestamp, time_frame=self.time_frame)
        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)
        stop_timestamp = floor_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)

        # shift forward first timestamp by train_length and val_length
        shift_length = self.train_length + self.val_length
        first_timestamp = int(first_timestamp + shift_length * self.time_frame)
        first_timestamp = floor_timestamp(timestamp=first_timestamp, time_frame=self.time_frame)
        first_datetime = datetime.fromtimestamp(first_timestamp)
        first_datetime = (first_datetime.replace(day=1) + timedelta(days=31)).replace(day=1)
        first_timestamp = first_datetime.timestamp()

        # sync start timestamp with first timestamp
        start_timestamp = max(first_timestamp, start_timestamp)
        folds_till_start_timestamp = (start_timestamp - first_timestamp) // self.test_length
        start_timestamp = first_timestamp + folds_till_start_timestamp * self.test_length

        self.__first_timestamp = int(first_timestamp)
        self.__last_timestamp = int(last_timestamp)
        self.__start_timestamp = int(start_timestamp)
        self.__stop_timestamp = int(min(last_timestamp, stop_timestamp))

        return self.__iter__()

    def __iter__(self):
        self.__samples_count = (self.stop_timestamp - self.start_timestamp) // int(self.time_frame)
        self.__folds_count = math.ceil(self.samples_count / self.test_length)
        self.__index = -1

        return self

    def __next__(self) -> Fold:
        self.__index += 1
        if self.index < self.folds_count:
            test_start_timestamp = int(self.start_timestamp + self.index * self.test_length * int(self.time_frame))
            test_stop_timestamp = int(test_start_timestamp + self.test_length * int(self.time_frame))
            test_start_timestamp = floor_timestamp(timestamp=test_start_timestamp, time_frame=self.time_frame)
            test_stop_timestamp = ceil_timestamp(timestamp=test_stop_timestamp, time_frame=self.time_frame)
            test_timestamps = list(range(test_start_timestamp, test_stop_timestamp, int(self.time_frame)))

            val_stop_timestamp = test_start_timestamp
            val_start_timestamp = int(val_stop_timestamp - self.val_length * int(self.time_frame))
            val_start_timestamp = floor_timestamp(timestamp=val_start_timestamp, time_frame=self.time_frame)
            val_stop_timestamp = floor_timestamp(timestamp=val_stop_timestamp, time_frame=self.time_frame)
            val_timestamps = list(range(val_start_timestamp, val_stop_timestamp, int(self.time_frame)))

            train_stop_timestamp = val_start_timestamp
            train_start_timestamp = int(train_stop_timestamp - self.train_length * int(self.time_frame))
            train_start_timestamp = floor_timestamp(timestamp=train_start_timestamp, time_frame=self.time_frame)
            train_stop_timestamp = floor_timestamp(timestamp=train_stop_timestamp, time_frame=self.time_frame)
            train_timestamps = list(range(train_start_timestamp, train_stop_timestamp, int(self.time_frame)))

            if 0 < len(set(train_timestamps).intersection(val_timestamps)):
                raise RuntimeError("Common timestamps found between the train and validation sets.")
            if 0 < len(set(train_timestamps).intersection(test_timestamps)):
                raise RuntimeError("Common timestamps found between the train and test sets.")
            if 0 < len(set(val_timestamps).intersection(test_timestamps)):
                raise RuntimeError("Common timestamps found between the validation and test sets.")

            fold = Fold(
                index=self.index,
                train_timestamps=train_timestamps,
                val_timestamps=val_timestamps,
                test_timestamps=test_timestamps
            )

            return fold
        else:
            raise StopIteration
