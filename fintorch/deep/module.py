import os.path
import pickle
from abc import ABC
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch

from .criterion import CE
from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
from .lr_scheduler import StepLR
from .model import Model
from .optimizer import Adam
from .trainer import Trainer
from ..dtype import Data
from ..setting import MODULE_DIR
from ..utils.directory import create_directory


class Module(ABC):
    def __init__(self, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__model: Model = model

        self.__last_train_timestamp: int = 0
        self.__trainer: Trainer = Trainer(
            epochs_count=20,
            cross_validation=CrossValidation(dataset=self.dataset),
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
        return os.path.join(
            MODULE_DIR, self.dataset.label_transform.symbol,
            str(int(self.dataset.label_transform.time_frame)),
            self.dataset.feature_transform.short_name,
            self.dataset.label_transform.short_name,
            self.model.short_name
        )

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"dim-sequence-{self.dataset.feature_transform.dim_sequence}.pkl")

    def optimize_and_store(self, timestamps: List[int]) -> Generator[Fold, None, None]:
        try:
            with open(self.path, "rb") as file:
                timestamps_to_fold = pickle.load(file)
        except (FileNotFoundError, EOFError) as e:
            timestamps_to_fold = {}

        # train on missed timestamps if exists
        missed_timestamps = [ts for ts in timestamps if ts not in timestamps_to_fold]
        if 0 < len(missed_timestamps):
            generator = self.trainer.optimize(dataset=self.dataset, model=self.model, timestamps=missed_timestamps)
            for fold in generator:
                if fold.done:
                    timestamps_to_fold.update({ts: fold for ts in fold.test_timestamps})

                yield fold

            create_directory(self.directory)
            with open(self.path, "wb+") as file:
                pickle.dump(timestamps_to_fold, file)

    def predict(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        try:
            with open(self.path, "rb") as file:
                timestamps_to_fold: Dict[int, Fold] = pickle.load(file)
        except (FileNotFoundError, EOFError) as e:
            self.optimize_and_store(timestamps=timestamps)
            with open(self.path, "rb") as file:
                timestamps_to_fold: Dict[int, Fold] = pickle.load(file)

        y_hat = []
        for timestamp in timestamps:
            model_state_dict = timestamps_to_fold[timestamp].best_val_epoch.model_state_dict
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

            feature = [item for item in self.dataset.feature_transform.transform_sf(data=data, timestamps=[timestamp])]

            x = torch.from_numpy(np.array(feature))
            y_hat = self.model(x)

        return torch.cat(y_hat, dim=0)

    def _post_load_fn(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y

    def __str__(self):
        return "{} {} {} {} {}" \
            .format(self.dataset.label_transform.symbol, self.dataset.label_transform.time_frame,
                    self.dataset.feature_transform.short_name, self.dataset.label_transform.short_name,
                    self.model.short_name)
