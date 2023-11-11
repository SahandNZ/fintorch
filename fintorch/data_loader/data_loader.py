from typing import Tuple

import torch

from fintorch.dataset.dataset import Dataset


class DataLoader:
    def __init__(self, batch_size: int, shuffle: bool = True, auto_cuda: bool = True):
        self.__batch_size: int = batch_size
        self.__shuffle: bool = shuffle
        self.__auto_cuda: bool = auto_cuda

        self._dataset: Dataset = None
        self._index: int = None

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def shuffle(self) -> bool:
        return self.__shuffle

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def device(self) -> str:
        return torch.device('cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu')

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def index(self) -> int:
        return self._index

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self):
        self._index = -1
        if self.shuffle:
            self._dataset.shuffle()

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self._index * self.batch_size < len(self._dataset.x):
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size

            batch_x = self._dataset.x[start_index: stop_index]
            batch_y = self._dataset.y[start_index: stop_index]

            return batch_x, batch_y
        else:
            raise StopIteration
