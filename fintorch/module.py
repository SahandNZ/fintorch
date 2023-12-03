import copy
import itertools
import os
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from fintorch.cross_validation.fold import Fold
from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.model.model import Model
from fintorch.trainer import Trainer
from fintorch.utils.directory import create_directory
from fintorch.utils.unpickler import Unpickler


class Module:
    def __init__(self, trainer: Trainer, dataset: Dataset, model: Model):
        self.__dataset: Dataset = dataset
        self.__trainer: Trainer = trainer
        self.__model: Model = copy.deepcopy(model)

        # properties
        self.__folds: List[Fold] = None
        self.__overall_fold: Fold = None

        # control fields
        self.__getstate_mode: str = None

    @staticmethod
    def create_and_save_modules(trainer: Trainer, datasets: List[Dataset], models: List[Model], root: str,
                                show_progress_bar: bool = True, print_logs: bool = True):
        bar = list(itertools.product(datasets, models))
        if show_progress_bar:
            bar = tqdm(bar, desc="Creating and saving module")

        for dataset, model in bar:
            Module.create_and_save(trainer=trainer, dataset=dataset, model=model, root=root, print_logs=print_logs)

    @staticmethod
    def create_and_save(trainer: Trainer, dataset: Dataset, model: Model, root: str, print_logs: bool = True):
        module = Module(trainer=trainer, dataset=dataset, model=model)

        if print_logs:
            print("-" * 32, module.short_name, "-" * 32)

        module.optimize()
        module.save()

        if print_logs:
            module.overall_fold.print_classification_logs()

        del module

    @staticmethod
    def load_modules(root: str) -> List:
        modules = []
        for item in os.listdir(root):
            path = os.path.join(root, item)
            if os.path.isfile(path):
                module = Module.load(path)
                modules.append(module)

        return modules

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

    @property
    def overall_fold(self) -> Fold:
        return self.__overall_fold

    @property
    def root_dir(self) -> str:
        data_root_dir = os.environ.get("DATA_ROOT", "./data")
        root_dir = os.path.join(data_root_dir, "module")
        create_directory(root_dir)

        return root_dir

    def path(self, mode: str) -> str:
        symbol = self.dataset.label_transform.symbol
        time_frame = self.dataset.label_transform.time_frame
        module_root_dir = os.path.join(self.root_dir, mode, symbol, str(time_frame))
        create_directory(module_root_dir)
        path = os.path.join(module_root_dir, f'{self.short_name}.pkl')

        return path

    def optimize(self):
        self.__folds, self.__overall_fold = self.trainer.optimize(dataset=self.dataset, model=self.model)
        self.model.load_state_dict(self.folds[-1].best_test_metrics.model_state_dict)

    def predict(self, data: Data, timestamps: List[int], show_progress_bar: bool) -> np.array:
        x = self.dataset.preprocess(data=data, timestamps=timestamps, show_progress_bar=show_progress_bar)
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model(x).cpu().numpy()

        return y_hat

    def save(self, mode: str = "experiment"):
        self.__getstate_mode = mode
        self.__dataset.getstate_mode = mode
        self.__overall_fold.getstate_mode = mode

        with open(self.path(mode=mode), 'wb+') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str, auto_cuda: bool = True):
        with open(path, 'rb') as file:
            unpickler = Unpickler(file=file, auto_cuda=auto_cuda)
            module = unpickler.load()

        return module

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct["_Module__root"] = None
        if "deployment" == self.__getstate_mode:
            if "_Module__folds" in dct:
                del dct["_Module__folds"]

        return dct
