from typing import List

import numpy as np
import torch
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BatsfDataset(Dataset):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        super().__init__(feature_transform, label_transform)

    def prepare(self, data: Data, samples_count: int = 0, progress: Progress = None):
        sampling_time_frame = min(max(self.feature_transform.time_frames), self.label_transform.time_frame)
        timestamps = data[data.symbols[0], sampling_time_frame].index.to_list()

        labels_dict = self.label_transform.fit_transform(data=data, timestamps=timestamps)
        timestamps = list(labels_dict.keys())[-samples_count:]
        features_dict = self.feature_transform.fit_transform(data=data, timestamps=timestamps, progress=progress)

        start_timestamp, x, y = None, [], []
        for timestamp in timestamps:
            if timestamp in features_dict and timestamp in labels_dict:
                start_timestamp = start_timestamp or timestamp
                x.append(features_dict[timestamp])
                y.append(labels_dict[timestamp])

        # convert x and y to tensor and slice dataframe
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()
        df = self.label_transform.dataframe.copy()
        df = df[start_timestamp <= df.index.to_series()]

        # set x, y, and df properties
        self.preset(x=x, y=y, df=df)

    def preprocess(self, data: Data, timestamps: List[int], progress: Progress = None) -> torch.Tensor:
        features_dict = self.feature_transform.fit_transform(data=data, timestamps=timestamps, progress=progress)
        features = features_dict.values()
        x = torch.from_numpy(np.array(features)).float()

        return x
