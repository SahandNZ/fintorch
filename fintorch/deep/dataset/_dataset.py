import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

from ..transform.feature import FeatureTransform
from ..transform.label import LabelTransform
from ...dtype import DataCollection
from ...enum import TimeFrame
from ...utils.timestamp import floor_timestamp, ceil_timestamp


class Dataset(ABC):
    def __init__(self):
        self._feature_transforms: Union[List[FeatureTransform], None] = None
        self._label_transform: Union[LabelTransform, None] = None
        self._static_hash: Union[int, None] = None

    @property
    def feature_transforms(self) -> List[FeatureTransform]:
        return self._feature_transforms

    @property
    def feature_transform(self) -> FeatureTransform:
        return self._feature_transforms[0]

    @property
    def label_transform(self) -> LabelTransform:
        return self._label_transform

    @property
    def symbol(self) -> str:
        return self.label_transform.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.label_transform.time_frame

    @property
    def symbols(self) -> List[str]:
        ft_symbols = [ft.symbol for ft in self.feature_transforms]
        lt_symbols = [self.label_transform.symbol]
        return list(set(ft_symbols + lt_symbols))

    @property
    def time_frames(self) -> List[TimeFrame]:
        ft_time_frames = [ft.time_frame for ft in self.feature_transforms]
        lt_time_frames = [self.label_transform.time_frame]
        return list(set(ft_time_frames + lt_time_frames))

    @property
    def static_hash(self) -> int:
        return self._static_hash

    @property
    def dim_input_time_frame(self) -> int:
        return len(self.feature_transforms)

    @property
    def dim_input_sequence(self) -> int:
        return self.feature_transforms[0].dim_sequence

    @property
    def dim_input_feature(self) -> int:
        return self.feature_transforms[0].dim_feature

    @property
    def dim_output_time_frame(self) -> int:
        return 1

    @property
    def dim_output_sequence(self) -> int:
        return self._label_transform.dim_sequence

    @property
    def dim_output_feature(self) -> int:
        return self._label_transform.dim_feature

    def open(self, read_only: bool = False) -> None:
        for feature_transform in self.feature_transforms:
            feature_transform.open(read_only=read_only)
        self.label_transform.open(read_only=read_only)

    def close(self) -> None:
        for feature_transform in self.feature_transforms:
            feature_transform.close()
        self.label_transform.close()

    def get_start_timestamp(self, dc: DataCollection) -> int:
        start_timestamp = -math.inf
        for feature_transform in self.feature_transforms:
            ft_start_timestamp = feature_transform.get_start_timestamp(dc=dc)
            start_timestamp = max(start_timestamp, ft_start_timestamp)

        return start_timestamp

    def get_stop_timestamp(self, dc: DataCollection) -> int:
        stop_timestamp = math.inf
        for feature_transform in self.feature_transforms:
            ft_stop_timestamp = feature_transform.get_stop_timestamp(dc=dc)
            stop_timestamp = min(stop_timestamp, ft_stop_timestamp)

        return stop_timestamp

    def get_timestamps(self, dc: DataCollection) -> List[int]:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)

        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)
        stop_timestamp = ceil_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)
        return list(range(start_timestamp, stop_timestamp, int(self.time_frame)))

    @abstractmethod
    def load_x(self, timestamps: List[int]) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def load_y(self, timestamps: List[int]) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def load(self, timestamps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return ("{} {} {} {}"
        .format(
            self.symbol,
            self.time_frame,
            self.feature_transforms[0].short_name,
            self.label_transform.short_name
        ))
