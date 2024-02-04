from abc import ABC
from typing import Generator, Tuple

import torch

from .criterion import CE
from .cross_validation import SlidingWindowCrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
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
            epochs_count=20,
            cross_validation=SlidingWindowCrossValidation(train_percentage=50, val_percentage=20, window_length=50000),
            data_loader=DataLoader(batch_size=1024, post_load_fn=self._post_load_fn),
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

    def optimize_and_store(self) -> Generator[Fold, None, None]:
        generator = self.trainer.optimize(dataset=self.dataset, model=self.model)
        for fold in generator:
            yield fold

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _post_load_fn(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y

    def __str__(self):
        return "{} {} {} {} {}" \
            .format(self.dataset.label_transform.symbol, self.dataset.label_transform.time_frame,
                    self.dataset.feature_transform.short_name, self.dataset.label_transform.short_name,
                    self.model.short_name)
