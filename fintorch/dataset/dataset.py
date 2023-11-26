import copy
import itertools
import os
import pickle
from abc import abstractmethod
from typing import List, Type

import pandas as pd
import torch

from fintorch.component import Component
from fintorch.data import Data
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.directory import create_directory


class Dataset(Component):
    def __init__(self, feature_transform: FeatureTransform, label_transform: LabelTransform):
        name = feature_transform.name + ' | ' + label_transform.name
        short_name = feature_transform.short_name + ' | ' + label_transform.short_name
        super().__init__(name=name, short_name=short_name, description="")

        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform

        # properties
        self.__x: torch.Tensor = None
        self.__y: torch.Tensor = None
        self.__df: pd.DataFrame = None

        # control flags
        self.__keep_x_in_getstate: bool = False

    @staticmethod
    def create_and_save_datasets(dataset: Type, feature_transforms: List[FeatureTransform],
                                 label_transforms: List[LabelTransform], data: Data, root: str, samples_count: int = -1,
                                 show_progress_bar: bool = True):
        for feature_transform, label_transform in list(itertools.product(feature_transforms, label_transforms)):
            Dataset.create_and_save(dataset, feature_transform, label_transform, data, root, samples_count,
                                    show_progress_bar)

    @staticmethod
    def create_and_save(dataset: Type, feature_transform: FeatureTransform, label_transform: LabelTransform, data: Data,
                        root: str, samples_count: int = -1, show_progress_bar: bool = False):
        d = dataset(feature_transform=feature_transform, label_transform=label_transform)
        d.prepare(data=data, samples_count=samples_count, show_progress_bar=show_progress_bar)
        d.save(root=root)
        del d

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            dataset = pickle.load(file)

        return dataset

    @staticmethod
    def aggregate(datasets: List):
        aggregated = datasets[0].copy()
        for dataset in datasets[1:]:
            aggregated.append(dataset, inplace=True)

        return aggregated

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    @property
    def x(self) -> torch.Tensor:
        return self.__x

    @x.deleter
    def x(self):
        del self.__x

    @property
    def y(self) -> torch.Tensor:
        return self.__y

    @y.deleter
    def y(self):
        del self.__y

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @df.deleter
    def df(self):
        del self.__df

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
        x = self.x.clone().detach()
        y = self.y.clone().detach()
        df = self.df.copy() if self.df is not None else None

        dataset = copy.deepcopy(self)
        dataset.preset(x=x, y=y, df=df)

        return dataset

    def shuffle(self):
        random_index = torch.randperm(len(self))
        x = self.x[random_index]
        y = self.y[random_index]
        self.preset(x=x, y=y, df=None)

    def to(self, device: str):
        self.__x = self.x.to(device)
        self.__y = self.y.to(device)

    @abstractmethod
    def prepare(self, data: Data, samples_count: int = -1, show_progress_bar: bool = False):
        raise NotImplemented()

    @abstractmethod
    def preprocess(self, data: Data, timestamps: List[int], show_progress_bar: bool = False) -> torch.Tensor:
        raise NotImplemented()

    def update(self, data: Data, show_progress_bar: bool = False):
        pass

    def append(self, other, inplace: bool = False):
        if self != other:
            raise ValueError("These two datasets cannot be concatenated.")

        samples_count = len(self) + other.samples_count
        x = torch.cat([self.x, other.x], dim=0)
        y = torch.cat([self.y, other.y], dim=0)
        df = pd.concat([self.df, other.df]) if self.df is not None and other.df is not None else None

        if inplace:
            self.preset(x=x, y=y, df=df)
        else:
            dataset = Dataset(feature_transform=self.feature_transform, label_transform=self.label_transform)
            dataset.preset(x=x, y=y, df=df)
            return dataset

    def save(self, root: str):
        if not os.path.exists(root):
            create_directory(root)
        path = os.path.join(root, f'{self.short_name}.pkl')
        with open(path, 'wb+') as file:
            self.__keep_x_in_getstate = True
            pickle.dump(self, file)
            self.__keep_x_in_getstate = False

        return path

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

            dataset = Dataset(feature_transform=self.feature_transform, label_transform=self.label_transform)
            dataset.preset(x=x, y=y, df=df)
            return dataset

        else:
            raise Exception("type of item must be int or slice.")

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other):
        return self.feature_transform == other.feature_transform and self.label_transform == other.label_transform

    def __getstate__(self):
        dct = dict(self.__dict__)
        if not self.__keep_x_in_getstate:
            del dct['_Dataset__x']
            dct['_Dataset__keep_x_in_getstate'] = False

        return dct
