import os.path
import pickle
from abc import ABC
from typing import Generator, List, Tuple

import numpy as np
import torch

from .criterion import CE
from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
from .lr_scheduler import LrScheduler
from .model import Model
from .optimizer import Optimizer
from .trainer import Trainer
from ..dtype import DataCollection
from ..setting import MODULE_DIR
from ..utils.directory import create_directory


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__model: Model = model

        self.__cross_validation = CrossValidation(interval=self.dataset.interval)
        self.__trainer: Trainer = Trainer(
            epochs_count=20,
            data_loader=DataLoader(batch_size=1024, post_load_fn=Module._post_load_fn),
            criterion=CE(),
            optimizer=Optimizer(torch_optimizer_type=torch.optim.Adam, lr=1e-3, weight_decay=1e-2),
            lr_scheduler=LrScheduler(torch_lr_scheduler_type=torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
            gradient_clipping_threshold=None,
        )

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def cross_validation(self) -> CrossValidation:
        return self.__cross_validation

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @property
    def directory(self) -> str:
        return os.path.join(
            MODULE_DIR,
            self.dataset.label_transform.symbol,
            str(int(self.dataset.label_transform.time_frame)),
            self.dataset.feature_transform.short_name,
            self.dataset.label_transform.short_name,
            self.model.short_name
        )

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"dim-sequence-{self.dataset.feature_transform.dim_sequence}.pkl")

    @staticmethod
    def _post_load_fn(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y

    def optimize(self, dc: DataCollection) -> Generator[Fold, None, None]:
        # safe load folds_dict
        try:
            with open(self.path, "rb") as file:
                folds_dict = pickle.load(file)
        except (FileNotFoundError, EOFError):
            folds_dict = {}

        # optimize new folds
        symbol_info = dc.get_symbol_info(symbol=self.dataset.label_transform.symbol)
        for fold in self.cross_validation(on_board_timestamp=symbol_info.on_board_timestamp):
            self.dataset.prepare(dc=dc, timestamps=fold.timestamps)
            key = (fold.test_start_timestamp, fold.test_stop_timestamp)
            if key not in folds_dict:
                folds_dict[key] = fold
                for _ in self.trainer.optimize_fold(dataset=self.dataset, model=self.model, fold=fold):
                    yield fold

        # update folds_dict
        create_directory(self.directory)
        with open(self.path, "wb+") as file:
            pickle.dump(folds_dict, file)

    def predict(self, dc: DataCollection, timestamps: List[int]) -> torch.Tensor:
        pass

    def __str__(self):
        return "{} {} {} {} {}" \
            .format(
            self.dataset.label_transform.symbol,
            self.dataset.label_transform.time_frame,
            self.dataset.feature_transform.short_name,
            self.dataset.label_transform.short_name,
            self.model.short_name
        )
