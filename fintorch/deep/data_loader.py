import copy
import math
import random
from typing import Callable, Iterator, Tuple, List

import numpy as np
import torch

from fintorch.deep.dtype import Dataset


class DataLoader:
    def __init__(self, batch_size: int, post_load_fn: Callable = None):
        self.__batch_size: int = batch_size
        self.__post_load_fn: Callable = post_load_fn

        self.__dataset: Dataset = None
        self.__timestamps: List[int] = None
        self.__batch_count: int = None
        self.__shuffle: bool = None

        self._index: int = -1

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def post_load_fn(self) -> Callable:
        return self.__post_load_fn

    @property
    def batch_count(self) -> int:
        return self.__batch_count

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
        return self._index

    def __call__(self, dataset: Dataset, timestamps: List[int], shuffle: bool) -> Iterator:
        self.__dataset = dataset
        self.__timestamps = copy.deepcopy(timestamps)
        self.__batch_count = math.ceil(len(timestamps) / self.batch_size)
        self.__shuffle = shuffle

        return self.__iter__()

    def __iter__(self):
        self._index = -1
        if self.shuffle:
            random.shuffle(self.timestamps)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self.index < self.batch_count:
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size
            batch_timestamps = self.timestamps[start_index: stop_index]
            batch_x, batch_y = self.dataset[batch_timestamps]

            if 0 < len(batch_x) and 0 < len(batch_y):
                if self.post_load_fn is not None:
                    batch_x, batch_y = self.post_load_fn(batch_x, batch_y)

                if torch.isnan(batch_x).max().item() or torch.isnan(batch_y).max().item():
                    raise RuntimeError("NaN in batch")
                if torch.isinf(batch_x).max().item() or torch.isinf(batch_y).max().item():
                    raise RuntimeError("INF in batch")

            return batch_x, batch_y
        else:
            raise StopIteration
