from abc import abstractmethod

import pandas as pd
import torch

from fintorch.transform.label.label_transform import LabelTransform
from fintorch.transform.transform import Transform


class Dataset:
    def __init__(self, feature_transform: Transform, label_transform: LabelTransform):
        self.__feature_transform: Transform = feature_transform
        self.__label_transform: LabelTransform = label_transform

        self.x: torch.Tensor = None
        self.y: torch.Tensor = None
        self.df: pd.DataFrame = None

    @property
    def name(self) -> str:
        return self.__feature_transform.name + '-' + self.__label_transform.name

    @property
    def feature_transform(self) -> Transform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    def preset(self, x: torch.Tensor = None, y: torch.Tensor = None, df: pd.DataFrame = None):
        self.x: torch.Tensor = x
        self.y: torch.Tensor = y
        self.df: pd.DataFrame = df

    @abstractmethod
    def prepare(self, df: pd.DataFrame):
        raise NotImplemented()

    @abstractmethod
    def preprocess(self, df: pd.DataFrame):
        raise NotImplemented()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if isinstance(item, slice):
            x = self.x[item]
            y = self.y[item] if self.y is not None else None
            df = self.df[item] if self.df is not None else None
            dataset = Dataset(feature_transform=self.feature_transform, label_transform=self.label_transform)
            dataset.preset(x, y, df)

            return dataset

        else:
            raise Exception("type of item must be slice.")
