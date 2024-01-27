from abc import ABC, abstractmethod
from typing import List

from rich.progress import Progress

from ..dtype import Dataset
from ..model import Model
from ..trainer import Trainer
from ...dtype import Data


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model, trainer: Trainer):
        self.__dataset: Dataset = dataset
        self.__model: Model = model
        self.__trainer: Trainer = trainer

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
    def predict(self, data: Data, timestamps: List[int]) -> List:
        raise NotImplementedError()
