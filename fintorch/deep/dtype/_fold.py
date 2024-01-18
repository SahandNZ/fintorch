from datetime import datetime
from typing import List


class Fold:
    def __init__(self, train_timestamps: List[int], validation_timestamps: List[int], test_timestamps: List[int]):
        self.__train_timestamps: List[int] = train_timestamps
        self.__validation_timestamps: List[int] = validation_timestamps
        self.__test_timestamp: List[int] = test_timestamps

    @property
    def train_timestamps(self) -> List[int]:
        return self.__train_timestamps

    @property
    def validation_timestamps(self) -> List[int]:
        return self.__validation_timestamps

    @property
    def test_timestamps(self) -> List[int]:
        return self.__test_timestamp

    @property
    def train_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.train_timestamps[0])

    @property
    def validation_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.validation_timestamps[0])

    @property
    def test_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[0])

    @property
    def test_stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[-1])


