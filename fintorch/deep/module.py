from abc import ABC, abstractmethod
from typing import Tuple

import torch
from rich.progress import Progress

from .criterion import CE
from .cross_validation import SlidingWindowCrossValidation
from .data_loader import DataLoader
from .dtype import Dataset
from .lr_scheduler import StepLR
from .model import Model
from .optimizer import Adam
from .trainer import Trainer


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__model: Model = model

        self.__last_train_timestamp: int = 0
        self.__trainer: Trainer = Trainer(
            cross_validation=SlidingWindowCrossValidation(train_percentage=80, val_percentage=10, window_length=10000),
            data_loader=DataLoader(post_load_fn=self._post_load_fn),
            criterion=CE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-2),
            lr_scheduler=StepLR(step_size=1, gamma=0.9),
            gradient_clipping_threshold=None,
        )

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
    def directory(self) -> str:
        raise NotImplementedError()

    @property
    def path(self) -> str:
        raise NotImplementedError()

    def optimize_and_store(self, epochs: int, batch_size: int, progress: Progress = None):
        folds = self.trainer.optimize(dataset=self.dataset, model=self.model, epochs=epochs, batch_size=batch_size,
                                      progress=progress)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _post_load_fn(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y
