import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fintorch.dataset.dataset import Dataset
from fintorch.transform.label.label_transform import LabelTransform
from fintorch.transform.transform import Transform


class BsfDataset(Dataset):
    def __init__(self, sequence_length: int, feature_transform: Transform, label_transform: LabelTransform):
        super().__init__(feature_transform, label_transform)
        self.__sequence_length: int = sequence_length

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    def prepare(self, df: pd.DataFrame):
        # create dataframes
        fdf = self.feature_transform.transform(df)
        ldf = self.label_transform.transform(df)

        # align dataframes
        start_timestamp = max([df.index.to_series().min() for df in [fdf, ldf]])
        fdf = fdf[start_timestamp <= fdf.index]
        ldf = ldf[start_timestamp <= ldf.index]

        # create samples
        x, y = [], []
        for index in tqdm(list(range(self.sequence_length, len(ldf)))):
            label = ldf.label.iloc[index]
            wdf = fdf.iloc[index - self.sequence_length: index]
            wdf = (wdf - wdf.mean()) / wdf.std()
            feature = wdf.to_numpy()

            y.append(label)
            x.append(feature)

        # slice dataframe and convert x and y to tensor
        self.df = ldf[self.sequence_length:]
        self.x = torch.from_numpy(np.array(x)).float()
        if 1 == self.label_transform.num_classes:
            self.y = torch.from_numpy(np.array(y)).float().unsqueeze(-1)
        else:
            self.y = torch.nn.functional.one_hot(torch.from_numpy(np.array(y)).long(), num_classes=-1).float()

    def preprocess(self, df: pd.DataFrame):
        fdf = self.feature_transform.transform(df)
        wdf = fdf.iloc[-self.sequence_length:]
        wdf = (wdf - wdf.mean()) / wdf.std()
        x = torch.from_numpy(np.array(wdf.to_numpy())).float().unsqueeze(0)

        return x
