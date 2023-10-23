from typing import List

import numpy as np
import torch

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BatsfDataset(Dataset):
    def __init__(self, samples_count: int, feature_transform: FeatureTransform, label_transform: LabelTransform,
                 show_progress_bar: bool):
        super().__init__(samples_count=samples_count, feature_transform=feature_transform,
                         label_transform=label_transform, show_progress_bar=show_progress_bar)

    def prepare(self, data: Data):
        # create dataframes
        feature_data = self.feature_transform.fit(data)
        label_dataframe = self.label_transform.fit(data)

        timestamps = label_dataframe.index.to_list()[-self.sample_count:]
        features = self.feature_transform.transform(data=feature_data, timestamps=timestamps,
                                                    show_progress_bar=self.show_progress_bar)
        labels = self.label_transform.transform(df=label_dataframe, timestamps=timestamps)

        x, y = [], []
        start_timestamp = timestamps[-1]
        for timestamp, feature, label in zip(timestamps, features, labels):
            if not np.isnan(feature).max() and not np.isnan(label).max():
                start_timestamp = min(start_timestamp, timestamp)
                x.append(feature)
                y.append(label)

        # convert x and y to tensor and slice dataframe
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()
        df = label_dataframe[start_timestamp <= label_dataframe.index.to_series()]

        # set x, y, and df properties
        self.preset(x=x, y=y, df=df)

    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool) -> torch.Tensor:
        feature_data = self.feature_transform.fit(data=data)
        features = self.feature_transform.transform(data=feature_data, timestamps=timestamps,
                                                    show_progress_bar=show_progress_bar)

        x = torch.from_numpy(np.array(features)).float()
        return x
