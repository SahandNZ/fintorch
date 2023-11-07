from typing import Tuple

import torch

from fintorch.dataset.dataset import Dataset
from fintorch.utils.memory import print_memory_status


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
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def index(self) -> int:
        return self._index

    def set_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def load(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.__auto_cuda and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        return x, y

    def __iter__(self):
        self._index = -1

        # shuffle part
        if self.shuffle:
            random_index = torch.randperm(len(self._dataset))
            x = self._dataset.x[random_index]
            y = self._dataset.y[random_index]
            self._dataset.preset(x=x, y=y, df=None)

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self._index * self.batch_size < len(self._dataset.x):
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size

            batch_x = self._dataset.x[start_index: stop_index]
            batch_y = self._dataset.y[start_index: stop_index]

            loaded_batch_x, loaded_batch_y = self.load(batch_x, batch_y)

            return loaded_batch_x, loaded_batch_y

        else:
            raise StopIteration
