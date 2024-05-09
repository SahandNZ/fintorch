import copy
import gc
import os.path
import pickle
import time
from abc import ABC
from typing import Any, Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold, Status
from .lr_scheduler import LrScheduler
from .model import Model
from .optimizer import Optimizer
from .trainer import Trainer
from .transform.feature import FeatureTransform
from .transform.label import LabelTransform
from ..dtype import DataCollection
from ..enum import TimeFrame
from ..settings import MODULE_DIR
from ..utils.directory import create_directory
from ..utils.function import call_with_dict
from ..utils.hash import static_list_hash
from ..utils.plot import draw_predictions
from ..utils.timestamp import to_timestamp, floor_timestamp, ceil_timestamp


class Module(ABC):
    def __init__(
            self,
            feature_transform: FeatureTransform,
            label_transform: LabelTransform,
            model_type: Type[Model],
            model_kwargs: Dict[str, Any],
            cross_validation_kwargs: Dict[str, Any],
            data_loader_kwargs: Dict[str, Any],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_kwargs: Dict[str, Any],
            trainer_kwargs: Dict[str, Any],
    ):
        self.__model_type: Type[Model] = model_type
        self.__model_kwargs: Dict[str, Any] = model_kwargs

        self.__dataset: Dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform)
        self.__model: Union[Model, None] = None
        self.__cross_validation = CrossValidation(time_frame=self.time_frame, **cross_validation_kwargs)
        self.__trainer: Trainer = Trainer(
            data_loader=DataLoader(post_load_fn=Module._post_load_fn, **data_loader_kwargs),
            optimizer=Optimizer(**optimizer_kwargs),
            lr_scheduler=LrScheduler(**lr_scheduler_kwargs),
            **trainer_kwargs
        )

        # calculate static_hash value
        model_static_hash: int = static_list_hash([
            self.__model_type.__name__,
            self.__model_kwargs["num_hidden_layers"],
            int(self.__model_kwargs["dropout"] * 10 ** 2),
            self.__model_kwargs["batch_norm"]
        ])
        self.__static_hash: int = static_list_hash([
            self.dataset.static_hash,
            self.cross_validation.static_hash,
            self.trainer.static_hash,
            model_static_hash
        ])

        # states
        self.__directory: Union[str, None] = None
        self.__open_mode: Union[str, None] = None
        self.__folds_dict: Union[Dict[Tuple[int, int], Fold], None] = None
        self.__y_hats_dict: Union[Dict[int, np.array], None] = None

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
        return self.dataset.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.dataset.time_frame

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    @property
    def directory(self) -> str:
        return self.__directory

    @property
    def folds_dict(self) -> Dict[Tuple[int, int], Fold]:
        return self.__folds_dict

    @property
    def y_hats_dict(self) -> Dict[int, np.array]:
        return {k: v for k, v in sorted(self.__y_hats_dict.items(), key=lambda item: item[0], reverse=True)}

    @staticmethod
    def _post_load_fn(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, y

    def state_dict(self) -> Dict:
        return {
            "folds_dict": self.folds_dict,
            "y_hats_dict": self.y_hats_dict
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.__folds_dict = state_dict.get("folds_dict", {})
        self.__y_hats_dict = state_dict.get("y_hats_dict", {})

    def open(self) -> None:
        # open dataset files from
        self.dataset.open()

        # create model
        self.__model_kwargs.update({
            "dim_input_sequence": self.dataset.feature_transform.dim_sequence,
            "dim_input_feature": self.dataset.feature_transform.dim_feature,
            "dim_output_sequence": self.dataset.label_transform.dim_sequence,
            "dim_output_feature": self.dataset.label_transform.dim_feature,
        })
        self.__model = call_with_dict(self.__model_type, self.__model_kwargs)

        # safe load state dict
        self.__directory = os.path.join(MODULE_DIR, str(self.static_hash))
        state_dict_path = os.path.join(self.directory, "state-dict.pkl")
        try:
            with open(state_dict_path, "rb") as file:
                state_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            state_dict = {}

        self.load_state_dict(state_dict=state_dict)

    def close(self) -> None:
        self.dataset.close()

        # store state dict
        create_directory(self.directory)
        state_dict_path = os.path.join(self.directory, "state-dict.pkl")
        with open(state_dict_path, "wb+") as file:
            pickle.dump(self.state_dict(), file)

        # remove model
        self.__model = None
        gc.collect()

    def get_start_timestamp(self, dc: DataCollection) -> float:
        first_timestamp = self.dataset.feature_transform.get_start_timestamp(dc=dc)
        last_timestamp = self.dataset.label_transform.get_stop_timestamp(dc=dc)

        tmp_cross_validation = copy.deepcopy(self.cross_validation)
        tmp_cross_validation(first_timestamp=first_timestamp, last_timestamp=last_timestamp)
        start_timestamp = tmp_cross_validation.start_timestamp
        del tmp_cross_validation

        return start_timestamp

    def get_stop_timestamp(self, dc: DataCollection) -> float:
        return self.dataset.feature_transform.get_stop_timestamp(dc=dc)

    def get_timestamps(
            self,
            dc: DataCollection,
            start_date: Union[str, None] = None,
            stop_date: Union[str, None] = None
    ) -> List[float]:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)

        if start_date is not None:
            start_timestamp = max(start_timestamp, int(to_timestamp(date=start_date)))
        if stop_date is not None:
            stop_timestamp = min(stop_timestamp, int(to_timestamp(date=stop_date)))

        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)
        stop_timestamp = ceil_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)
        return list(np.arange(start_timestamp, stop_timestamp, float(self.time_frame)))

    def optimize(
            self,
            dc: DataCollection,
            start_date: Union[str, None] = None,
            stop_date: Union[str, None] = None,
    ) -> Generator[Status, None, None]:
        # parse start_date
        start_timestamp = to_timestamp(date=start_date) if start_date is not None else None
        stop_timestamp = to_timestamp(date=stop_date) if stop_date is not None else None

        # setup cross validation
        first_timestamp = self.dataset.feature_transform.get_start_timestamp(dc=dc)
        last_timestamp = self.dataset.feature_transform.get_stop_timestamp(dc=dc)
        folds_iterator = self.cross_validation(
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp
        )

        # optimize new folds
        status = Status(total_folds_count=self.cross_validation.folds_count)
        for fold in folds_iterator:
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

    def predict(self, dc: DataCollection, timestamps: List[float], mode="val") -> Dict[float, np.array]:
        # load from y_hats_dict
        output = {ts: self.y_hats_dict[ts] for ts in timestamps if ts in self.y_hats_dict}
        output = {ts: y_hat for ts, y_hat in output.items() if not np.isnan(y_hat).max()}
        missing_timestamps = [ts for ts in timestamps if ts not in output]

        # predict missing timestamps
        if 0 < len(missing_timestamps):
            self.model.eval()
            with torch.no_grad():
                for key, fold in self.folds_dict.items():
                    self.model.load_state_dict(getattr(fold, f"best_{mode}_epoch").model_state_dict)
                    fold_timestamps = [ts for ts in missing_timestamps if key[0] <= ts <= key[1]]
                    if 0 < len(fold_timestamps):
                        x = self.dataset.preprocess(dc=dc, timestamps=fold_timestamps)
                        y_hats = self.model(x).numpy()
                        output.update({ts: y_hats[index] for index, ts in enumerate(fold_timestamps)})

            # update and sort y_hats_dict
            self.y_hats_dict.update(output)

        # set missed timestamps to array of np.nan values
        nan_y_hat = np.zeros((self.model.dim_output_sequence, self.model.dim_output_feature)) * np.nan
        missing_timestamps = [ts for ts in timestamps if ts not in output]
        output.update({ts: nan_y_hat for ts in missing_timestamps})
        sorted_output = {k: v for k, v in sorted(output.items(), key=lambda item: item[0])}

        return sorted_output

    def draw_ohlc_plot(
            self,
            dc: DataCollection,
            start_date: str,
            stop_date: str,
            mode: str = "val"
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        # draw ohlc and labels
        fig, ohlc_ax, df = self.dataset.label_transform.draw_ohlc_plot(
            dc=dc,
            start_date=start_date,
            stop_date=stop_date
        )

        # add prediction column to df
        y_hats_dict = self.predict(dc=dc, timestamps=df.index.to_list(), mode=mode)
        prediction_dict = {k: np.argmax(v) for k, v in y_hats_dict.items()}
        df["prediction"] = prediction_dict.values()

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
        if exc_type is None:
            self.close()

    def __str__(self):
        return (
            "{} {} {} {} {}"
            .format(
                self.dataset.symbol,
                self.dataset.time_frame,
                self.dataset.feature_transform.short_name,
                self.dataset.label_transform.short_name,
                self.__model_type.__name__
            )
        )
