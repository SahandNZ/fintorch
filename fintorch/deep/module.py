import gc
import itertools
import os.path
import pickle
import time
from abc import ABC
from typing import Any, Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .criterion import CE
from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold, Status
from .lr_scheduler import LrScheduler
from .model import MODEL_TYPES, Model
from .optimizer import Optimizer
from .trainer import Trainer
from .transform.feature import FEATURE_TRANSFORM_TYPES, FeatureTransform
from .transform.label import LABEL_TRANSFORM_TYPES, LabelTransform
from ..dtype import DataCollection
from ..enum import TimeFrame
from ..setting import INTERVAL, MODEL_KWARGS, MODULE_DIR, TRANSFORM_KWARGS, EPOCHS_COUNT, BATCH_SIZE, LR, WEIGHT_DECAY
from ..utils.args import DefaultNamespace
from ..utils.directory import create_directory
from ..utils.function import call_with_dict
from ..utils.plot import draw_predictions
from ..utils.timestamp import to_datetime


class Module(ABC):
    def __init__(
            self,
            dataset: Dataset,
            model_type: Type[Model],
            model_kwargs: Dict[str, Any] = MODEL_KWARGS,
            epochs_count: int = EPOCHS_COUNT,
            batch_size: int = BATCH_SIZE,
            lr: float = LR,
            weight_decay: float = WEIGHT_DECAY
    ):
        self.__dataset: Dataset = dataset

        self.__model_type: Type[Model] = model_type
        self.__model_kwargs: Dict[str, Any] = model_kwargs
        self.__model: Union[Model, None] = None

        self.__cross_validation = CrossValidation(interval=self.dataset.interval)
        self.__trainer: Trainer = Trainer(
            epochs_count=epochs_count,
            data_loader=DataLoader(batch_size=batch_size, post_load_fn=Module._post_load_fn),
            criterion=CE(),
            optimizer=Optimizer(torch_optimizer_type=torch.optim.Adam, lr=lr, weight_decay=weight_decay),
            lr_scheduler=LrScheduler(torch_lr_scheduler_type=torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
            gradient_clipping_threshold=None,
        )

        self.__directory: str = None
        self.__folds_dict_path: str = None
        self.__folds_dict: Dict[Tuple[int, int], Fold] = None

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
        return self.__folds_dict_path

    @property
    def folds_dict(self) -> Dict[Tuple[int, int], Fold]:
        return self.__folds_dict

    @staticmethod
    def _post_load_fn(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = y.squeeze(-2)
        return x, y

    def open(self) -> None:
        # open dataset files from disk and create model
        self.dataset.open()
        self.__model = call_with_dict(self.__model_type, self.__model_kwargs)

        # assign values to directory and folds_dict_path
        self.__directory = os.path.join(
            MODULE_DIR,
            str(hash(self.dataset)),
            str(hash(self.model)),
            str(hash(self.cross_validation)),
            str(hash(self.trainer))
        )
        self.__folds_dict_path = os.path.join(self.directory, "folds-dict.pkl")

        # safe load self.folds_dict
        try:
            with open(self.folds_dict_path, "rb") as file:
                self.__folds_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__folds_dict = {}

    def close(self) -> None:
        self.dataset.close()

        # dump self.fold_dict
        create_directory(self.directory)
        with open(self.folds_dict_path, "wb+") as file:
            completed_folds_dict = {k: v for k, v in self.folds_dict.items() if v.epochs_count == len(v.epochs)}
            pickle.dump(completed_folds_dict, file)

        # remove model
        self.__model = None
        gc.collect()

    def get_test_start_timestamp(self, dc: DataCollection) -> int:
        first_valid_timestamp = self.dataset.feature_transform.get_first_valid_timestamp(dc=dc)
        self.cross_validation(first_valid_timestamp=first_valid_timestamp)
        return self.cross_validation.test_start_timestamp

    def get_test_timestamps(self, dc: DataCollection) -> List[int]:
        test_start_timestamps = self.get_test_start_timestamp(dc=dc)
        feature_last_valid_timestamps = self.dataset.feature_transform.get_last_valid_timestamp(dc=dc)
        return list(range(test_start_timestamps, feature_last_valid_timestamps, self.time_frame))

    def get_vaid_test_timestamps(self, dc: DataCollection) ->List[int]:
        test_start_timestamps = self.get_test_start_timestamp(dc=dc)
        label_last_valid_timestamps = self.dataset.label_transform.get_last_valid_timestamp(dc=dc)
        return list(range(test_start_timestamps, label_last_valid_timestamps, self.time_frame))

    def optimize(self, dc: DataCollection, start_date: Union[str, None] = None) -> Generator[Status, None, None]:
        # parse start_date
        start_date = to_datetime(date=start_date) if start_date is not None else None

        # setup cross validation
        first_valid_timestamp = self.dataset.feature_transform.get_first_valid_timestamp(dc=dc)
        iterator = self.cross_validation(first_valid_timestamp=first_valid_timestamp)

        # optimize new folds
        status = Status(folds_count=self.cross_validation.folds_count)
        for fold in iterator:
            # skip folds which are not include in start_timestamp
            if start_date is not None and fold.test_stop_timestamp < start_date.timestamp():
                continue

            start_time = time.time()
            key = (fold.test_start_timestamp, fold.test_stop_timestamp)
            if key in self.folds_dict:
                fold = self.folds_dict[key]
                status.append_fold(fold=fold)
                elapsed_time = time.time() - start_time
                status.update_elapsed_time(elapsed_time=elapsed_time)
                yield status
            else:
                self.folds_dict[key] = fold
                status.append_fold(fold=fold)
                for _ in self.trainer.optimize_fold(dataset=self.dataset, model=self.model, fold=fold):
                    elapsed_time = time.time() - start_time
                    status.update_elapsed_time(elapsed_time=elapsed_time)
                    yield status

    def predict(self, dc: DataCollection, timestamps: List[int], mode="val") -> Dict[int, List[float]]:
        # predict timestamps
        y_hats_dict = {}
        self.model.eval()
        with torch.no_grad():
            for key, fold in self.folds_dict.items():
                self.model.load_state_dict(getattr(fold, f"best_{mode}_epoch").model_state_dict)
                fold_timestamps = [ts for ts in timestamps if key[0] <= ts <= key[1]]
                if 0 < len(fold_timestamps):
                    x = self.dataset.preprocess(dc=dc, timestamps=fold_timestamps)
                    y_hats = self.model(x).tolist()
                    y_hats_dict.update({ts: y_hats[index] for index, ts in enumerate(fold_timestamps)})

        # set missed timestamps to None
        missed_timestamps = [ts for ts in timestamps if ts not in y_hats_dict]
        y_hats_dict.update({ts: None for ts in missed_timestamps})

        return y_hats_dict

    def draw_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str, mode: str = "val") \
            -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        # draw ohlc and labels
        fig, ohlc_ax, df = self.dataset.label_transform.draw_ohlc_plot(
            dc=dc,
            start_date=start_date,
            stop_date=stop_date
        )

        # add prediction column to df
        y_hats_dict = self.predict(dc=dc, timestamps=df.index.to_list(), mode=mode)
        df["prediction"] = [np.argmax(value) for value in y_hats_dict.values()]

        # draw predictions
        df.reset_index(drop=False, inplace=True)
        draw_predictions(ohlc_ax=ohlc_ax, df=df)
        df.set_index("timestamp", inplace=True)

        return fig, ohlc_ax, df

    def show_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str, mode: str = "val") -> None:
        _, ohlc_ax, _ = self.draw_ohlc_plot(dc=dc, start_date=start_date, stop_date=stop_date, mode=mode)

        ohlc_ax.grid()
        ohlc_ax.legend()

        plt.title("{} (from {} to {})".format(str(self), start_date, stop_date))
        plt.show()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self):
        return "{} {} {} {} {}" \
            .format(
            self.dataset.label_transform.symbol,
            self.dataset.label_transform.time_frame,
            self.dataset.feature_transform.short_name,
            self.dataset.label_transform.short_name,
            self.__model_type.__name__
        )


def create_module(
        feature_transform_type: Type[FeatureTransform],
        label_transform_type: Type[LabelTransform],
        model_type: Type[Model],
        transform_kwargs: Dict[str, Any] = TRANSFORM_KWARGS,
        model_kwargs: Dict[str, Any] = MODEL_KWARGS,
        interval: TimeFrame = INTERVAL
) -> Module:
    # create feature and label transforms
    feature_transform = call_with_dict(feature_transform_type, transform_kwargs)
    label_transform = call_with_dict(label_transform_type, transform_kwargs)

    # create dataset
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=interval)

    # create module
    return Module(dataset=dataset, model_type=model_type, model_kwargs=model_kwargs)


def create_modules(args: DefaultNamespace) -> List[Module]:
    modules = []
    items = itertools.product(
        args.symbols,
        args.time_frames,
        FEATURE_TRANSFORM_TYPES,
        LABEL_TRANSFORM_TYPES,
        MODEL_TYPES
    )
    for symbol, time_frame, ft_type, lt_type, model_type in items:
        transform_kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}

        module = create_module(
            feature_transform_type=ft_type,
            label_transform_type=lt_type,
            model_type=model_type,
            transform_kwargs=transform_kwargs,
            model_kwargs=args.model_kwargs,
            interval=args.interval
        )

        modules.append(module)

    return modules
