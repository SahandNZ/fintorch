from typing import List, Tuple, Type, Dict

import numpy as np
import torch

from ._dataset import Dataset
from ..transform.feature import FeatureTransform
from ..transform.label import LabelTransform
from ...dtype import DataCollection
from ...enum import TimeFrame
from ...utils.function import call_with_dict
from ...utils.hash import static_list_hash


class TimeFrameSequenceFeatureDataset(Dataset):
    def __init__(
            self,
            feature_symbol: str,
            feature_time_frames: List[TimeFrame],
            feature_transform_kwargs: Dict,
            feature_transform_type: Type[FeatureTransform],
            label_symbol: str,
            label_time_frame: TimeFrame,
            label_transform_kwargs: Dict,
            label_transform_type: Type[LabelTransform],
    ) -> None:
        super().__init__()

        feature_transforms = []
        ft_kwargs = feature_transform_kwargs.copy()
        for feature_time_frame in feature_time_frames:
            ft_kwargs.update({"symbol": feature_symbol, "time_frame": feature_time_frame})
            feature_transform = call_with_dict(feature_transform_type, ft_kwargs)
            feature_transforms.append(feature_transform)

        lt_kwargs = label_transform_kwargs.copy()
        lt_kwargs.update({"symbol": label_symbol, "time_frame": label_time_frame})
        label_transform = call_with_dict(label_transform_type, lt_kwargs)

        static_hash = static_list_hash([
            *[feature_transform.static_hash for feature_transform in feature_transforms],
            label_transform.static_hash
        ])

        self._feature_transforms: List[FeatureTransform] = feature_transforms
        self._label_transform: LabelTransform = label_transform
        self._static_hash: int = static_hash

    def preprocess(self, dc: DataCollection, timestamps: List[int]) -> torch.Tensor:
        features = []
        for feature_transform in self.feature_transforms:
            time_frame_features = list(feature_transform.transform_sf(dc=dc, timestamps=timestamps))
            features.append(time_frame_features)

        x = torch.from_numpy(np.array(features)).transpose(0, 1).float()

        return x

    def load(self, timestamps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        for feature_transform in self.feature_transforms:
            time_frame_features = feature_transform.load_sf(timestamps=timestamps)
            features.append(time_frame_features)
        labels = self.label_transform.load_sf(timestamps=timestamps)

        # convert to torch.tensor
        x = torch.from_numpy(np.array(features)).transpose(0, 1).float()
        y = torch.from_numpy(np.array(labels)).float()
        
        if 0 < len(y):
            y = y.squeeze(1)

        return x, y
