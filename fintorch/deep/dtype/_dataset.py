from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import torch

from ...deep.transform.feature import FeatureTransform
from ...deep.transform.label import LabelTransform
from ...dtype import DataCollection
from ...enum import TimeFrame
from ...utils.hash import static_list_hash


class Dataset(ABC):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform

        self.__static_hash: int = static_list_hash([
            self.feature_transform.static_hash,
            self.label_transform.static_hash,
        ])

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    @property
    def symbol(self) -> str:
        return self.label_transform.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.label_transform.time_frame

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def open(self) -> None:
        self.feature_transform.open()
        self.label_transform.open()

    def close(self) -> None:
        self.feature_transform.close()
        self.label_transform.close()

    def preprocess(self, dc: DataCollection, timestamps: List[float]) -> torch.Tensor:
        feature_generator = self.feature_transform.transform_sf(dc=dc, timestamps=timestamps)
        features = [feature for feature in feature_generator]
        x = torch.from_numpy(np.array(features)).float()

        return x

    def prepare(self, dc: DataCollection, timestamps: List[float]) -> None:
        self.feature_transform.prepare_sf(dc=dc, timestamps=timestamps)
        self.label_transform.prepare_sf(dc=dc, timestamps=timestamps)

    def load(self, timestamps: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_transform.load_sf(timestamps=timestamps)
        labels = self.label_transform.load_sf(timestamps=timestamps)

        # convert to torch.tensor
        x = torch.from_numpy(np.array(features)).float()
        y = torch.from_numpy(np.array(labels)).float()

        return x, y

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __getitem__(self, item: Union[float, List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(item, float):
            return self.load(timestamps=[item])
        elif isinstance(item, list):
            return self.load(timestamps=item)
        else:
            raise ValueError("item parameter must be float (single timestamp) or list of floats (multiple timestamps).")
