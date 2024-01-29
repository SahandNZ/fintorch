from abc import ABC, abstractmethod
from typing import Tuple

import torch
from rich.progress import Progress

from .dtype import Dataset
from .model import Model
from .trainer import Trainer


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model, trainer: Trainer):
        self.__dataset: Dataset = dataset
        self.__model: Model = model
        self.__trainer: Trainer = trainer

        self.__last_train_timestamp: int = 0

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
        folds = self.trainer.optimize(dataset=self.dataset, model=self.model, epochs=epochs, batch_size=batch_size,
                                      progress=progress)

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _post_load_fn(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
