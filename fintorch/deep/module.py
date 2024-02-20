import copy
import gc
import itertools
import os.path
import pickle
from abc import ABC
from typing import Any, Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .criterion import CE
from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
from .lr_scheduler import LrScheduler
from .model import MODEL_TYPES, Model
from .optimizer import Optimizer
from .trainer import Trainer
from .transform.feature import FEATURE_TRANSFORM_TYPES, FeatureTransform
from .transform.label import LABEL_TRANSFORM_TYPES, LabelTransform
from ..dtype import DataCollection
from ..enum import TimeFrame
from ..setting import SAMPLING_INTERVAL, MODEL_KWARGS, MODULE_DIR, TRANSFORM_KWARGS
from ..utils.directory import create_directory
from ..utils.function import call_with_dict
from ..utils.plot import draw_predictions


class Module(ABC):
    def __init__(
            self,
            dataset: Dataset,
            model_type: Type[Model],
            model_kwargs: Dict[str, Any],
            epoch: int = 20,
            batch_size: int = 1024,
            lr: float = 1e-3,
            weight_decay: float = 1e-2
    ):
        self.__dataset: Dataset = dataset

        self.__model_type: Type[Model] = model_type
        self.__model_kwargs: Dict[str, Any] = model_kwargs
        self.__model: Union[Model, None] = None

        self.__cross_validation = CrossValidation(interval=self.dataset.interval)
        self.__trainer: Trainer = Trainer(
            epochs_count=epoch,
            data_loader=DataLoader(batch_size=batch_size, post_load_fn=Module._post_load_fn),
            criterion=CE(),
            optimizer=Optimizer(torch_optimizer_type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
            lr_scheduler=LrScheduler(torch_lr_scheduler_type=torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
            gradient_clipping_threshold=None,
        )

        self.__directory = os.path.join(
            MODULE_DIR,
            str(hash(self.dataset)),
            str(hash(self.model)),
            str(hash(self.trainer))
        )

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    @property
    def model(self) -> Model:
        if self.__model is None:
            self.__model = call_with_dict(self.__model_type, self.__model_kwargs)

        return self.__model

    @model.deleter
    def model(self):
        self.__model = None
        gc.collect()

    @property
    def cross_validation(self) -> CrossValidation:
        return self.__cross_validation

    @property
    def trainer(self) -> Trainer:
        return self.__trainer

    @property
    def symbol(self) -> str:
        return self.dataset.label_transform.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.dataset.label_transform.time_frame

    @property
    def directory(self) -> str:
        return self.__directory

    @property
    def folds_dict_path(self) -> str:
        return os.path.join(self.directory, "folds-dict.pkl")

    @property
    def y_hats_dict_path(self) -> str:
        return os.path.join(self.directory, "y-hats-dict.pkl")

    @staticmethod
    def _post_load_fn(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y

    def optimize(self, dc: DataCollection) -> Generator[Fold, None, None]:
        # safe load folds_dict
        try:
            with open(self.folds_dict_path, "rb") as file:
                folds_dict = pickle.load(file)
        except (FileNotFoundError, EOFError):
            folds_dict = {}

        # optimize new folds
        for fold in self.cross_validation(start_timestamp=self.dataset.feature_transform.get_start_timestamp(dc=dc)):
            self.dataset.prepare(dc=dc, timestamps=fold.timestamps)
            key = (fold.test_start_timestamp, fold.test_stop_timestamp)
            if key not in folds_dict:
                folds_dict[key] = fold
                for _ in self.trainer.optimize_fold(dataset=self.dataset, model=self.model, fold=fold):
                    yield fold

        # remove model from memory to reduce memory usage
        del self.model

        # update folds_dict
        create_directory(self.directory)
        with open(self.folds_dict_path, "wb+") as file:
            pickle.dump(folds_dict, file)

    def predict(self, dc: DataCollection, timestamps: List[int], mode="val") -> Dict[int, List[float]]:
        # safe load folds_dict
        try:
            with open(self.folds_dict_path, "rb") as file:
                folds_dict = pickle.load(file)
        except (FileNotFoundError, EOFError):
            folds_dict = {}

        # predict timestamps
        y_hats_dict = {}
        self.model.eval()
        with torch.no_grad():
            for key, fold in folds_dict.items():
                self.model.load_state_dict(getattr(fold, f"best_{mode}_epoch").model_state_dict)
                fold_timestamps = [ts for ts in timestamps if key[0] <= ts <= key[1]]
                if 0 < len(fold_timestamps):
                    x = self.dataset.preprocess(dc=dc, timestamps=fold_timestamps)
                    y_hats = self.model(x).tolist()
                    y_hats_dict.update({ts: y_hats[index] for index, ts in enumerate(fold_timestamps)})

        # remove model to reduce memory usage
        del self.model

        # set missed timestamps to None
        missed_timestamps = [ts for ts in timestamps if ts not in y_hats_dict]
        y_hats_dict.update({ts: None for ts in missed_timestamps})

        return y_hats_dict

    def draw_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str) \
            -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        # draw ohlc and labels
        fig, ohlc_ax, df = self.dataset.label_transform.draw_ohlc_plot(
            dc=dc,
            start_date=start_date,
            stop_date=stop_date
        )

        # add prediction column to df
        y_hats_dict = self.predict(dc=dc, timestamps=df.index.to_list())
        df["prediction"] = [np.argmax(value) for value in y_hats_dict.values()]

        # draw predictions
        df.reset_index(drop=False, inplace=True)
        draw_predictions(ohlc_ax=ohlc_ax, df=df)
        df.set_index("timestamp", inplace=True)

        return fig, ohlc_ax, df

    def show_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str) -> None:
        _, ohlc_ax, _ = self.draw_ohlc_plot(dc=dc, start_date=start_date, stop_date=stop_date)

        ohlc_ax.grid()
        ohlc_ax.legend()

        plt.title("{} (from {} to {})".format(str(self), start_date, stop_date))
        plt.show()

    def __str__(self):
        return "{} {} {} {} {}" \
            .format(
            self.dataset.label_transform.symbol,
            self.dataset.label_transform.time_frame,
            self.dataset.feature_transform.short_name,
            self.dataset.label_transform.short_name,
            self.model.short_name
        )


def create_module(
        feature_transform_type: Type[FeatureTransform],
        label_transform_type: Type[LabelTransform],
        transform_kwargs: Dict[str, Any],
        model_type: Type[Model],
        model_kwargs: Dict[str, Any],
        interval: TimeFrame
) -> Module:
    # create feature and label transforms
    feature_transform = call_with_dict(feature_transform_type, transform_kwargs)
    label_transform = call_with_dict(label_transform_type, transform_kwargs)

    # create dataset
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=interval)

    # create module
    return Module(dataset=dataset, model_type=model_type, model_kwargs=model_kwargs)


def create_modules(symbols: List[str], time_frames: List[int]) -> List[Module]:
    modules = []
    items = itertools.product(symbols, time_frames, FEATURE_TRANSFORM_TYPES, LABEL_TRANSFORM_TYPES, MODEL_TYPES)
    for symbol, time_frame, lt_type, ft_type, model_type in items:
        transform_kwargs = copy.deepcopy(TRANSFORM_KWARGS)
        transform_kwargs = transform_kwargs.update({"symbol": symbol, "time_frame": time_frame})
        module = create_module(
            feature_transform_type=ft_type,
            label_transform_type=lt_type,
            transform_kwargs=transform_kwargs,
            model_type=model_type,
            model_kwargs=MODEL_KWARGS,
            interval=SAMPLING_INTERVAL
        )

        modules.append(module)

    return modules
