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


class SequenceFeatureDataset(Dataset):
    def __init__(
            self,
            feature_symbol: str,
            feature_time_frame: TimeFrame,
            feature_transform_kwargs: Dict,
            feature_transform_type: Type[FeatureTransform],
            label_time_frame: TimeFrame,
            label_symbol: str,
            label_transform_kwargs: Dict,
            label_transform_type: Type[LabelTransform],
    ) -> None:
        super().__init__()
        ft_kwargs = feature_transform_kwargs.copy()
        ft_kwargs.update({"symbol": feature_symbol, "time_frame": feature_time_frame})
        feature_transform = call_with_dict(feature_transform_type, ft_kwargs)

        lt_kwargs = label_transform_kwargs.copy()
        lt_kwargs.update({"symbol": label_symbol, "time_frame": label_time_frame})
        label_transform = call_with_dict(label_transform_type, lt_kwargs)

        static_hash = static_list_hash([feature_transform.static_hash, label_transform.static_hash])

        self._feature_transforms: List[FeatureTransform] = [feature_transform]
        self._label_transform: LabelTransform = label_transform
        self._static_hash: int = static_hash

        self.__feature_transform: FeatureTransform = feature_transform

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    def preprocess(self, dc: DataCollection, timestamps: List[int]) -> torch.Tensor:
        features = list(self.feature_transform.transform_sf(dc=dc, timestamps=timestamps))
        x = torch.from_numpy(np.array(features)).float()

        return x

    def load(self, timestamps: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_transform.load_sf(timestamps=timestamps)
        labels = self.label_transform.load_sf(timestamps=timestamps)

        # convert to torch.tensor
        x = torch.from_numpy(np.array(features)).float()
        y = torch.from_numpy(np.array(labels)).float()

        return x, y
