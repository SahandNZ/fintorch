from typing import List

import torch

from fintorch.dataset.batsf_dataset import BatsfDataset
from fintorch.dataset.data import Data
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BsfDataset(BatsfDataset):
    def __init__(self, samples_count: int, feature_transform: FeatureTransform, label_transform: LabelTransform,
                 show_progress_bar: bool):
        super().__init__(samples_count=samples_count, feature_transform=feature_transform,
                         label_transform=label_transform, show_progress_bar=show_progress_bar)

    def prepare(self, data: Data):
        super().prepare(data=data)

        x = self.x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)
        self.preset(x=x, y=self.y, df=self.df)

    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool) -> torch.Tensor:
        x = super().preprocess(data=data, timestamps=timestamps, show_progress_bar=show_progress_bar)
        x = x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)

        return x
