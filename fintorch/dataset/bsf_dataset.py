import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BsfDataset(Dataset):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform,
                 show_progress_bar: bool = False):
        super().__init__(feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=show_progress_bar)

    def prepare(self, df: pd.DataFrame):
        # create dataframes
        df = df.copy()
        df = self.feature_transform.fit(df)
        df = self.label_transform.fit(df)

        bar = range(self.feature_transform.sequence_length, len(df))
        if self.show_progress_bar:
            bar = tqdm(list(bar))

        # create samples
        x, y = [], []
        for index in bar:
            timestamp = df.index.to_series().iloc[index]
            label = self.label_transform.transform(df=df, timestamp=timestamp)
            feature = self.feature_transform.transform(df=df, timestamp=timestamp)

            y.append(label)
            x.append(feature)

        # slice dataframe and convert x and y to tensor
        df = df[self.feature_transform.sequence_length:]
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()

        # set x, y, and df properties
        self.preset(x=x, y=y, df=df)

    def preprocess(self, df: pd.DataFrame, timestamp: int) -> torch.Tensor:
        df = df.copy()
        df = self.feature_transform.fit(df=df)
        feature = self.feature_transform.transform(df=df, timestamp=timestamp)
        x = torch.from_numpy(feature).unsqueeze(0).float()

        return x
