from abc import ABC, abstractmethod
from typing import List

import torch
from rich.progress import Progress

from ..dtype import Dataset
from ..model import Model
from ..trainer import Trainer
from ...dtype import Data


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model, trainer: Trainer, auto_cuda: bool):
        self.__dataset: Dataset = dataset
        self.__model: Model = model
        self.__trainer: Trainer = trainer
        self.__auto_cuda: bool = auto_cuda

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() and self.auto_cuda else "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    @property
    @abstractmethod
    def directory(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def path(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def optimize_and_store(self, epochs: int, batch_size: int, progress: Progress = None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def preprocess(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        raise NotImplementedError()
