from typing import List

import torch

from fintorch.data import Data
from fintorch.dataset.batsf_dataset import BatsfDataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BsfDataset(BatsfDataset):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        super().__init__(feature_transform=feature_transform, label_transform=label_transform)

    def prepare(self, data: Data, samples_count: int = 0, show_progress_bar: bool = False):
        super().prepare(data=data, samples_count=samples_count, show_progress_bar=show_progress_bar)

        x = self.x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)
        self.preset(x=x, y=self.y, df=self.df)

    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool = False) -> torch.Tensor:
        x = super().preprocess(data=data, timestamps=timestamps, show_progress_bar=show_progress_bar)
        x = x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)

        return x
