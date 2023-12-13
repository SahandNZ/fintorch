from typing import List

import numpy as np
import torch

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BatsfDataset(Dataset):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        super().__init__(feature_transform, label_transform)

    def prepare(self, data: Data, samples_count: int = 0, show_progress_bar: bool = False):
        sampling_time_frame = min(max(self.feature_transform.time_frames), self.label_transform.time_frame)
        timestamps = data[data.symbols[0], sampling_time_frame].index.to_list()

        timestamp_to_label = self.label_transform.fit_transform(data=data, timestamps=timestamps)
        timestamps = list(timestamp_to_label.keys())[-samples_count:]
        timestamp_to_feature = self.feature_transform.fit_transform(data, timestamps, show_progress_bar)

        start_timestamp, x, y = None, [], []
        for timestamp in timestamps:
            if timestamp in timestamp_to_feature and timestamp in timestamp_to_label:
                start_timestamp = start_timestamp or timestamp
                x.append(timestamp_to_feature[timestamp])
                y.append(timestamp_to_label[timestamp])

        # convert x and y to tensor and slice dataframe
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()
        df = self.label_transform.dataframe.copy()
        df = df[start_timestamp <= df.index.to_series()]

        # set x, y, and df properties
        self.preset(x=x, y=y, df=df)

    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool = False) -> torch.Tensor:
        features = self.feature_transform.fit_transform(data, timestamps, show_progress_bar)
        x = torch.from_numpy(np.array(features)).float()

        return x
