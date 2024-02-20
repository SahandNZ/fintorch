from abc import ABC
from typing import List, Tuple, Union
from datetime import datetime
import numpy as np
import torch

from fintorch.deep.transform.feature import FeatureTransform
from fintorch.deep.transform.label import LabelTransform
from fintorch.dtype import DataCollection
from fintorch.enum import TimeFrame
from fintorch.utils.hash import static_list_hash


class Dataset(ABC):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform, interval: TimeFrame):
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform
        self.__interval: TimeFrame = interval

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    def preprocess(self, dc: DataCollection, timestamps: List[int]) -> torch.Tensor:
        feature_generator = self.feature_transform.transform_sf(dc=dc, timestamps=timestamps)
        features = [feature for feature in feature_generator]
        x = torch.from_numpy(np.array(features)).float()

        return x

    def prepare(self, dc: DataCollection, timestamps: List[int]) -> None:
        features_generator = self.feature_transform.transform_sf(dc=dc, timestamps=timestamps)
        labels_generators = self.label_transform.transform_sf(dc=dc, timestamps=timestamps)

        for feature, label in zip(features_generator, labels_generators):
            pass

    def _load_samples(self, timestamps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        features_generator = self.feature_transform.load_sf(timestamps=timestamps)
        labels_generator = self.label_transform.load_sf(timestamps=timestamps)

        valid_features, valid_labels = [], []
        for feature, label in zip(features_generator, labels_generator):
            if feature is not None and label is not None:
                valid_features.append(feature)
                valid_labels.append(label)

        x = torch.from_numpy(np.array(valid_features)).float()
        y = torch.from_numpy(np.array(valid_labels)).float()

        return x, y

    def __getitem__(self, item: Union[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(item, int):
            return self._load_samples(timestamps=[item])
        elif isinstance(item, list):
            return self._load_samples(timestamps=item)
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")

    def __hash__(self):
        return static_list_hash(
            [
                int(self.interval),
                self.label_transform.symbol,
                self.label_transform.time_frame,
                self.feature_transform.short_name,
                self.label_transform.short_name
            ]
        )
