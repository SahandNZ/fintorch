from abc import ABC
from datetime import datetime
from typing import List, Tuple, Union
from datetime import datetime
import numpy as np
import torch

from fintorch.deep.transform.feature import FeatureTransform
from fintorch.deep.transform.label import LabelTransform
from fintorch.dtype import Data
from fintorch.enum import TimeFrame
from fintorch.utils.timestamp import create_timestamps


class Dataset(ABC):
    def __init__(self, start_date: str, stop_date: str, interval: TimeFrame, feature_transform: FeatureTransform,
                 label_transform: LabelTransform):
        self.__start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.__stop_date = datetime.strptime(stop_date, "%Y-%m-%d")
        self.__interval: TimeFrame = interval
        self.__timestamps: List[int] = create_timestamps(self.start_date, self.stop_date, self.interval)

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
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    def prepare(self, data: Data, timestamps: List[int]) -> None:
        raise NotImplementedError()

    def preprocess(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        feature_generator = self.feature_transform.transform_sf(data=data, timestamps=timestamps)
        features = [feature for feature in feature_generator]
        x = torch.from_numpy(np.array(features)).float()

        return x

    def _load_samples(self, timestamps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_transform.load_sf(timestamps=timestamps)
        labels = self.label_transform.load_sf(timestamps=timestamps)

        valid_features, valid_labels = [], []
        for feature, label in zip(features, labels):
            if feature is not None and label is not None:
                valid_features.append(feature)
                valid_labels.append(label)

        x = torch.from_numpy(np.array(valid_features)).float()
        y = torch.from_numpy(np.array(valid_labels)).float()

        return x, y

    def __len__(self):
        return int(self.stop_date.timestamp() - self.start_date.timestamp()) // self.interval

    def __getitem__(self, item: Union[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(item, int):
            return self._load_samples(timestamps=[item])
        elif isinstance(item, list):
            return self._load_samples(timestamps=item)
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")
