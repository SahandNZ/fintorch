import copy
import math
import random
from datetime import datetime
from typing import Callable, Iterator, Tuple, List, Union

import torch
import numpy as np

from .dataset import Dataset
from .error import NanValueInBatchError
from ..utils.hash import static_hash


class DataLoader:
    def __init__(self, batch_size: int = None, batch_count: int = None):
        self.__batch_size: int = batch_size
        self.__batch_count: int = batch_count

        self.__dataset: Union[Dataset, None] = None
        self.__timestamps: Union[List[int], None] = None
        self.__drop_nan: Union[bool, None] = None
        self.__shuffle: Union[bool, None] = None

        self.__index: Union[int, None] = None
        self.__weight: Union[torch.Tensor, None] = None

        self.__static_hash: int = static_hash(self.batch_size or self.batch_count)

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def batch_count(self) -> int:
        return self.__batch_count

    @property
    def drop_nan(self) -> bool:
        return self.__drop_nan
    
    @property
    def shuffle(self) -> bool:
        return self.__shuffle

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def timestamps(self) -> List[int]:
        return self.__timestamps

    @property
    def index(self) -> int:
        return self.__index

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def __call__(self, dataset: Dataset, timestamps: List[int], drop_nan: bool, shuffle: bool) -> Iterator:
        self.__dataset = dataset
        self.__timestamps = copy.deepcopy(timestamps)
        self.__batch_count = self.batch_count or math.ceil(len(timestamps) / self.batch_size)
        self.__batch_size = self.batch_size or math.ceil(len(timestamps) / self.batch_count)
        self.__drop_nan = drop_nan
        self.__shuffle = shuffle

        return self.__iter__()

    def __iter__(self):
        self.__index = -1
        _, y = self.dataset[self.timestamps]
        y = y[~torch.any(y.isnan(), dim=1)]
        self.__weight = len(y) / (2 * y.sum(dim=0))

        if self.shuffle:
            random.shuffle(self.timestamps)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__index += 1

        if self.index < self.batch_count:
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size
            batch_timestamps = self.timestamps[start_index: stop_index]
            batch_x, batch_y = self.dataset[batch_timestamps]

            if 0 < len(batch_x) and 0 < len(batch_y):
                if self.drop_nan:
                    indices = ~torch.any(batch_y.squeeze(1).isnan(), dim=1)
                    batch_x, batch_y = batch_x[indices], batch_y[indices]

                if len(batch_x) != len(batch_y):
                    raise Exception("X and Y must have the same lengths.")

                if torch.any(batch_x.isnan()) or torch.any(batch_x.isinf()):
                    first_nan_datetime = datetime.fromtimestamp(batch_timestamps[0])
                    last_nan_datetime = datetime.fromtimestamp(batch_timestamps[-1])
                    message = "NaN or Inf values found in the features of batch data. (first: {} last:{})" \
                        .format(first_nan_datetime, last_nan_datetime)
                    raise NanValueInBatchError(message)

                if torch.any(batch_y.isnan()) or torch.any(batch_y.isinf()):
                    first_nan_datetime = datetime.fromtimestamp(batch_timestamps[0])
                    last_nan_datetime = datetime.fromtimestamp(batch_timestamps[-1])
                    message = "NaN or Inf values found in the labels of batch data. (first: {} last:{})" \
                        .format(first_nan_datetime, last_nan_datetime)
                    raise NanValueInBatchError(message)

            return batch_x, batch_y, self.__weight
        else:
            raise StopIteration
