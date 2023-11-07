import copy
from abc import abstractmethod
from typing import List

import pandas as pd
import torch

from fintorch.data import Data
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class Dataset:
    def __init__(self, samples_count: int, feature_transform: FeatureTransform = None,
                 label_transform: LabelTransform = None, show_progress_bar: bool = False):
        self.__samples_count: int = samples_count
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform
        self.__show_progress_bar: bool = show_progress_bar

        self.__x: torch.Tensor = None
        self.__y: torch.Tensor = None
        self.__df: pd.DataFrame = None

    @staticmethod
    def aggregate(datasets: List):
        aggregated = datasets[0].copy()
        for dataset in datasets[1:]:
            aggregated.append(dataset, inplace=True)

        return aggregated

    @property
    def name(self) -> str:
        return self.feature_transform.name + ' | ' + self.label_transform.name

    @property
    def short_name(self) -> str:
        return self.feature_transform.short_name + ' | ' + self.label_transform.short_name

    @property
    def samples_count(self) -> int:
        return self.__samples_count

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

    @property
    def need_preparation(self) -> bool:
        return self.x is None

    def reset(self):
        self.__x: torch.Tensor = None
        self.__y: torch.Tensor = None
        self.__df: pd.DataFrame = None

    def preset(self, x: torch.Tensor = None, y: torch.Tensor = None, df: pd.DataFrame = None):
        self.__x: torch.Tensor = x
        self.__y: torch.Tensor = y
        self.__df: pd.DataFrame = df

    def copy(self):
        dataset = copy.deepcopy(self)

        x = self.x.clone().detach()
        y = self.y.clone().detach()
        df = self.df.copy()
        dataset.preset(x=x, y=y, df=df)

        return dataset

    @abstractmethod
    def prepare(self, data: Data):
        raise NotImplemented()

    @abstractmethod
    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool) -> torch.Tensor:
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
            dataset = Dataset(samples_count=len(y), feature_transform=self.feature_transform,
                              label_transform=self.label_transform, show_progress_bar=self.show_progress_bar)
            dataset.preset(x, y, df)

            return dataset

        else:
            raise Exception("type of item must be int or slice.")

    def append(self, other, inplace: bool = False):
        if self != other:
            raise ValueError("These two datasets cannot be concatenated.")

        samples_count = self.samples_count + other.samples_count
        x = torch.cat([self.x, other.x], dim=0)
        y = torch.cat([self.y, other.y], dim=0)
        df = pd.concat([self.df, other.df])

        if not inplace:
            dataset = Dataset(samples_count=samples_count, feature_transform=self.feature_transform,
                              label_transform=self.label_transform, show_progress_bar=self.show_progress_bar)
            dataset.preset(x=x, y=y, df=df)
            return dataset
        else:
            self.__samples_count = samples_count
            self.__x = x
            self.__y = y
            self.__df = df

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        return self.feature_transform == other.feature_transform and self.label_transform == other.label_transform

    def __getstate__(self):
        dct = dict(self.__dict__)
        dct['_Dataset__x'] = None

        return dct
