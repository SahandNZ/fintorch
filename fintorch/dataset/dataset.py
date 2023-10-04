from abc import abstractmethod

import pandas as pd
import torch

from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class Dataset:
    def __init__(self, feature_transform: FeatureTransform = None, label_transform: LabelTransform = None,
                 show_progress_bar: bool = False):
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform
        self.__show_progress_bar: bool = show_progress_bar

        self.__x: torch.Tensor = None
        self.__y: torch.Tensor = None
        self.__df: pd.DataFrame = None

    @property
    def name(self) -> str:
        return self.feature_transform.short_name + ' | ' + self.label_transform.short_name

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    @property
    def show_progress_bar(self) -> bool:
        return self.__show_progress_bar

    @property
    def x(self) -> torch.Tensor:
        return self.__x

    @property
    def y(self) -> torch.Tensor:
        return self.__y

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def reset(self):
        self.__x: torch.Tensor = None
        self.__y: torch.Tensor = None
        self.__df: pd.DataFrame = None

    def preset(self, x: torch.Tensor = None, y: torch.Tensor = None, df: pd.DataFrame = None):
        self.__x: torch.Tensor = x
        self.__y: torch.Tensor = y
        self.__df: pd.DataFrame = df

    @abstractmethod
    def prepare(self, df: pd.DataFrame):
        raise NotImplemented()

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, timestamp: int):
        raise NotImplemented()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        if isinstance(item, int):
            x = self.x[item]
            y = self.y[item]

            return x, y

        elif isinstance(item, slice):
            x = self.x[item]
            y = self.y[item]
            df = self.df[item] if self.df is not None else None
            dataset = Dataset(feature_transform=self.feature_transform, label_transform=self.label_transform,
                              show_progress_bar=self.show_progress_bar)
            dataset.preset(x, y, df)

            return dataset

        else:
            raise Exception("type of item must be int or slice.")

    def __str__(self) -> str:
        return self.name
