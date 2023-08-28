import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fintorch.dataset.dataset import Dataset
from fintorch.transform.label.label_transform import LabelTransform
from fintorch.transform.transform import Transform


class BcwhDataset(Dataset):
    def __init__(self, width: int, height: int, feature_transform: Transform, label_transform: LabelTransform):
        super().__init__(feature_transform, label_transform)
        self.__width: int = width
        self.__height: int = height

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

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
        for index in tqdm(list(range(self.width * self.height, len(ldf)))):
            label = ldf.label.iloc[index]
            feature = []

            wdf = fdf.iloc[index - self.width * self.height: index]
            wdf = (wdf - wdf.mean()) / wdf.std()
            for window_index in range(0, self.width * self.height, self.width):
                feature.append(wdf.iloc[window_index: window_index + self.width].to_numpy())

            y.append(label)
            x.append(feature)

        # slice dataframe and convert x and y to tensor
        self.df = ldf[self.width * self.height:]
        self.x = torch.from_numpy(np.array(x)).float().unsqueeze(1)
        if 1 == self.label_transform.num_classes:
            self.y = torch.from_numpy(np.array(y)).float().unsqueeze(-1)
        else:
            self.y = torch.nn.functional.one_hot(torch.from_numpy(np.array(y)).long(), num_classes=-1).float()

    def preprocess(self, df: pd.DataFrame):
        fdf = self.feature_transform.transform(df)

        x = []
        wdf = fdf.iloc[-self.width * self.height:]
        wdf = (wdf - wdf.mean()) / wdf.std()
        for window_index in range(0, self.width * self.height, self.width):
            x.append(wdf.iloc[window_index: window_index + self.width].to_numpy())

        x = torch.from_numpy(np.array(x)).float().unsqueeze(0).unsqueeze(1)

        return x
