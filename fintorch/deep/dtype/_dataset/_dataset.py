from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

import torch
from rich.progress import Progress

from .._sample import Sample
from ...transform.feature import FeatureTransform
from ...transform.label import LabelTransform
from ....dtype import Data
from ....enum import TimeFrame
from ....utils.timestamp import create_timestamps


class Dataset(ABC):
    def __init__(self, start_date: str, stop_date: str, interval: TimeFrame, sequence_length: int,
                 feature_transform: FeatureTransform, label_transform: LabelTransform):
        self.__start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.__stop_date = datetime.strptime(stop_date, "%Y-%m-%d")
        self.__interval: TimeFrame = interval
        self.__timestamps: List[int] = create_timestamps(self.start_date, self.stop_date, self.interval)

        self.__sequence_length: int = sequence_length
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform

    @property
    def start_date(self) -> datetime:
        return self.__start_date

    @property
    def stop_date(self) -> datetime:
        return self.__stop_date

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    @property
    def timestamps(self) -> List[int]:
        return self.__timestamps

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    @abstractmethod
    def prepare(self, data: Data, progress: Progress = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _load_samples(self, timestamps: List[int]) -> List[Sample]:
        raise NotImplementedError()

    def __len__(self):
        return int(self.stop_date.timestamp() - self.start_date.timestamp()) // self.interval

    def __getitem__(self, item: Union[int, List[int]]) -> List[Sample]:
        if isinstance(item, int):
            return self._load_samples(timestamps=[item])
        elif isinstance(item, list):
            return self._load_samples(timestamps=item)
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")
