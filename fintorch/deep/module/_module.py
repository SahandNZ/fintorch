from abc import ABC, abstractmethod
from typing import List

from ..dtype import Dataset
from ..trainer import Trainer


class Module(ABC):
    def __init__(self, trainer: Trainer):
        super().__init__()
        self.__trainer: Trainer = trainer

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @abstractmethod
    def optimize(self, dataset: Dataset, epochs: int, batch_size: int):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset: Dataset, timestamps: List[int]) -> List:
        raise NotImplementedError()
