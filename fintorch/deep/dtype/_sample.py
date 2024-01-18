from datetime import datetime

import numpy as np


class Sample:
    def __init__(self, timestamp: int, feature: np.array, label: np.array):
        self.__timestamp: int = timestamp
        self.__datetime: datetime = datetime.fromtimestamp(timestamp)
        self.__feature: np.array = feature
        self.__label: np.array = label

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    def datetime(self) -> datetime:
        return self.__datetime

    @property
    def feature(self) -> np.array:
        return self.__feature

    @property
    def label(self) -> np.array:
        return self.__label

    @property
    def is_valid(self) -> bool:
        return self.feature is not None and self.label is not None
