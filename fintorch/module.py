import os
import pickle
from typing import List, Tuple

import numpy as np
import torch

from fintorch.cross_validation.fold import Fold
from fintorch.dataset.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.model.model import Model
from fintorch.trainer import Trainer
from fintorch.utils import create_directory


class Module:
    def __init__(self, trainer: Trainer, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__trainer: Trainer = trainer
        self.__model: Model = model

        self.__folds: List[Fold] = None

    @property
    def name(self) -> str:
        return self.dataset.name + ' | ' + self.model.name

    @property
    def short_name(self) -> str:
        return self.dataset.short_name + ' | ' + self.model.short_name

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def folds(self) -> List[Fold]:
        return self.__folds

    def optimize(self, data: Data):
        if self.dataset.x is None:
            self.dataset.prepare(data=data)
        self.__folds = self.trainer.optimize(dataset=self.dataset, model=self.model)
        self.model.load_state_dict(self.folds[-1].best_test_metrics.model_state_dict)

    def predict(self, data: Data, timestamps: int, show_progress_bar: bool) -> Tuple[np.array, np.array]:
        x, y = self.dataset.preprocess(data=data, timestamps=timestamps, show_progress_bar=show_progress_bar)
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model(x).cpu().numpy()

        return y.numpy(), y_hat

    def save(self, root: str) -> str:
        # mute unnecessary data from dataset
        x = self.dataset.x
        y = self.dataset.y
        df = self.dataset.df
        self.dataset.reset()

        # save module
        if not os.path.exists(root):
            create_directory(root)
        path = os.path.join(root, f'{self.name}.pkl')
        with open(path, 'wb+') as file:
            pickle.dump(self, file)

        # reset muted data to dataset
        self.dataset.preset(x=x, y=y, df=df)
        return path

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as file:
            loaded_module = pickle.load(file)

        return loaded_module
