import copy
import random
from typing import Callable, Iterator, Tuple, List

import numpy as np
import torch

from fintorch.deep.dtype import Dataset


class DataLoader:
    def __init__(self, post_load_fn: Callable = None):
        self.__post_load_fn: Callable = post_load_fn

        self.__dataset: Dataset = None
        self.__timestamps: List[int] = None
        self.__batch_size: int = None
        self.__shuffle: bool = None

        self._index: int = -1

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def shuffle(self) -> bool:
        return self.__shuffle

    @property
    def post_load_fn(self) -> Callable:
        return self.__post_load_fn

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def timestamps(self) -> List[int]:
        return self.__timestamps

    @property
    def index(self) -> int:
        return self._index

    def __call__(self, dataset: Dataset, timestamps: List[int], batch_size: int, shuffle: bool = True) -> Iterator:
        self.__dataset = dataset
        self.__timestamps = copy.deepcopy(timestamps)
        self.__batch_size = batch_size
        self.__shuffle = shuffle

        return self.__iter__()

    def __iter__(self):
        self._index = -1
        if self.shuffle:
            random.shuffle(self.timestamps)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self.index * self.batch_size < len(self.timestamps):
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size

            batch_timestamps = self.timestamps[start_index: stop_index]
            batch_samples = self.dataset[batch_timestamps]
            batch_valid_samples = [sample for sample in batch_samples if sample is not None and sample.is_valid]
            batch_x = torch.from_numpy(np.array([sample.feature for sample in batch_valid_samples])).float()
            batch_y = torch.from_numpy(np.array([sample.label for sample in batch_valid_samples])).float()

            if self.post_load_fn is not None:
                batch_x, batch_y = self.post_load_fn(batch_x, batch_y)

            return batch_x, batch_y
        else:
            raise StopIteration
