from typing import List

import torch
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.batsf_dataset import BatsfDataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BsfDataset(BatsfDataset):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        super().__init__(feature_transform=feature_transform, label_transform=label_transform)

    def prepare(self, data: Data, samples_count: int = 0, progress: Progress = None):
        super().prepare(data=data, samples_count=samples_count, progress=progress)

        x = self.x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)
        self.preset(x=x, y=self.y, df=self.df)

    def preprocess(self, data: Data, timestamps: List[int], progress: Progress = None) -> torch.Tensor:
        x = super().preprocess(data=data, timestamps=timestamps, progress=progress)
        x = x.permute(0, 3, 1, 2, 4)
        x = torch.flatten(x, start_dim=2)

        return x
