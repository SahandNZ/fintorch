from abc import abstractmethod, ABC
from typing import Tuple

import torch

from fintorch.dataset.dataset import Dataset


class DataLoader(ABC):
    def __init__(self, batch_size: int, shuffle: bool, auto_cuda: bool):
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

    @abstractmethod
    def __iter__(self):
        raise NotImplemented()

    @abstractmethod
    def __next__(self):
        raise NotImplemented()
