import os
import pickle

import pandas as pd
import torch

from fintorch.dataset.dataset import Dataset
from fintorch.model.model import Model
from fintorch.trainer import Trainer


class Module:
    def __init__(self, trainer: Trainer, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__trainer: Trainer = trainer
        self.__model: Model = model

    @property
    def name(self) -> str:
        return self.trainer.name + ' - ' + self.dataset.name + ' - ' + self.model.name

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def model(self) -> Model:
        return self.__model

    def optimize(self, df: pd.DataFrame):
        self.dataset.prepare(df)
        last_fold = self.trainer.optimize(dataset=self.dataset, model=self.model)[-1]
        self.model.load_state_dict(last_fold.best_test_metrics.model_state_dict)

    def predict(self, df: pd.DataFrame, timestamp: int) -> torch.Tensor:
        self.model.eval()
        x = self.dataset.preprocess(df=df, timestamp=timestamp)
        y_hat = self.model(x).cpu()

        return y_hat

    def save(self, root: str) -> str:
        # mute unnecessary data from dataset
        x = self.dataset.x
        y = self.dataset.y
        df = self.dataset.df
        self.dataset.reset()

        # save module
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
