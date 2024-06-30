import gc
import time
import copy
import pickle
import os.path
import filelock
from abc import ABC
from datetime import datetime
from typing import Any, Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dataset import Dataset
from .lr_scheduler import LrScheduler
from .metrics import Metrics
from .model import Model
from .optimizer import Optimizer
from .status import Fold, Status
from .trainer import Trainer
from ..dtype import DataCollection
from ..enum import TimeFrame
from ..settings import FINTORCH_MODULE_DIR
from ..utils.directory import create_directory
from ..utils.function import call_with_dict
from ..utils.hash import static_list_hash
from ..utils.plot import draw_predictions_based_on_labels
from ..utils.timestamp import to_timestamp, floor_timestamp, ceil_timestamp
from ..utils.memory import get_memory_status


class Module(ABC):
    def __init__(
            self,
            dataset: Dataset,
            model_type: Type[Model],
            model_kwargs: Dict[str, Any],
            cross_validation_kwargs: Dict[str, Any],
            data_loader_kwargs: Dict[str, Any],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_kwargs: Dict[str, Any],
            trainer_kwargs: Dict[str, Any],
    ):
        data_loader_kwargs.update(dataset.label_transform.data_loader_kwargs)
        optimizer_kwargs.update(dataset.label_transform.optimizer_kwargs)
        lr_scheduler_kwargs.update(dataset.label_transform.lr_scheduler_kwargs)
        trainer_kwargs.update(dataset.label_transform.trainer_kwargs)

        self.__model_type: Type[Model] = model_type
        self.__model_kwargs: Dict[str, Any] = model_kwargs

        self.__dataset: Dataset = dataset
        self.__cross_validation = CrossValidation(time_frame=self.time_frame, **cross_validation_kwargs)
        self.__trainer: Trainer = Trainer(
            data_loader=DataLoader(**data_loader_kwargs),
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
        self.__directory: str = os.path.join(FINTORCH_MODULE_DIR, str(self.static_hash))
        self.__folds_dict_path: str = os.path.join(self.directory, "folds-dict.pkl")
        self.__y_hats_dict_path: str = os.path.join(self.directory, "y-hats-dict.pkl")

        self.__folds_dict: Union[Dict[Tuple[int, int], Fold], None] = None
        self.__y_hats_dict: Union[Dict[int, np.array], None] = None
        self.__rewrite_folds_dict: bool = False
        self.__rewrite_y_hats_dict: bool = False

    @property
    def dataset(self) -> Dataset:
        return self.__dataset
    
    @property
    def model_type(self) -> Type[Model]:
        return self.__model_type

    @property
    def model(self) -> Model:
        return getattr(self, "__model")

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
    def folds_dict_path(self) -> str:
        return self.__folds_dict_path
    
    @property
    def y_hats_dict_path(self) -> str:
        return self.__y_hats_dict_path

    @property
    def folds_dict(self) -> Dict[Tuple[int, int], Fold]:
        return self.__folds_dict

    @property
    def y_hats_dict(self) -> Dict[int, np.array]:
        return {k: v for k, v in sorted(self.__y_hats_dict.items(), key=lambda item: item[0], reverse=True)}

    def open(self) -> None:
        # open dataset files from
        self.dataset.open()

        # create model
        self.__model_kwargs.update({
            "dim_input_time_frame": self.dataset.dim_input_time_frame,
            "dim_input_sequence": self.dataset.dim_input_sequence,
            "dim_input_feature": self.dataset.dim_input_feature,

            "dim_output_time_frame": self.dataset.dim_output_time_frame,
            "dim_output_sequence": self.dataset.dim_output_sequence,
            "dim_output_feature": self.dataset.dim_output_feature,
        })
        
        model = call_with_dict(self.__model_type, self.__model_kwargs)
        setattr(self, "__model", model)

        # safe load folds-dict and y-hats-dict
        try:
            with filelock.FileLock(self.folds_dict_path):
                with open(self.folds_dict_path, "rb") as file:
                    self.__folds_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            self.__folds_dict = {}
            
        try:
            with filelock.FileLock(self.y_hats_dict_path):
                with open(self.y_hats_dict_path, "rb") as file:
                    self.__y_hats_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__y_hats_dict = {}
            
        self.__rewrite_folds_dict = False
        self.__rewrite_y_hats_dict = False

    def close(self) -> None:
        self.dataset.close()

        # store folds dict and y hats dict
        if self.__rewrite_folds_dict:
            create_directory(self.directory)
            with filelock.FileLock(self.folds_dict_path):
                with open(self.folds_dict_path, "wb+") as file:
                    pickle.dump(self.folds_dict, file)

        if self.__rewrite_y_hats_dict:
            create_directory(self.directory)
            with filelock.FileLock(self.y_hats_dict_path):
                with open(self.y_hats_dict_path, "wb+") as file:
                    pickle.dump(self.y_hats_dict, file)


        # clear states to free allocated memory
        delattr(self, "__model")
        self.__folds_dict.clear()
        self.__y_hats_dict.clear()

        gc.collect()
        torch.cuda.empty_cache()

    def get_start_timestamp(self, dc: DataCollection) -> float:
        first_timestamp = self.dataset.get_start_timestamp(dc=dc)
        last_timestamp = self.dataset.get_stop_timestamp(dc=dc)

        tmp_cross_validation = copy.deepcopy(self.cross_validation)
        tmp_cross_validation(first_timestamp=first_timestamp, last_timestamp=last_timestamp)
        start_timestamp = tmp_cross_validation.start_timestamp
        del tmp_cross_validation

        return start_timestamp

    def get_stop_timestamp(self, dc: DataCollection) -> float:
        return self.dataset.get_stop_timestamp(dc=dc)

    def get_timestamps(
            self,
            dc: DataCollection,
            start_date: Union[str, None] = None,
            stop_date: Union[str, None] = None
    ) -> List[int]:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)

        if start_date is not None:
            start_timestamp = int(max(start_timestamp, int(to_timestamp(date=start_date))))
        if stop_date is not None:
            stop_timestamp = int(min(stop_timestamp, int(to_timestamp(date=stop_date))))

        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)
        stop_timestamp = ceil_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)
        return list(range(start_timestamp, stop_timestamp, int(self.time_frame)))

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
        first_timestamp = self.dataset.get_start_timestamp(dc=dc)
        last_timestamp = self.dataset.get_stop_timestamp(dc=dc)
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
                self.__rewrite_folds_dict = True
                
                status.append_fold(fold=fold)
                for _ in self.trainer.optimize_fold(dataset=self.dataset, model=self.model, fold=fold):
                    elapsed_time = time.time() - start_time
                    status.update_elapsed_time(elapsed_time=elapsed_time)
                    yield status
                    
    def _predict(self, dc: DataCollection, timestamps: List[int], mode="val") -> Dict[float, np.array]:
        # move model to cuda device if its available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device)
        
        # generate predictions
        result = {}
        with torch.no_grad():
            for key, fold in self.folds_dict.items():
                self.model.load_state_dict(getattr(fold, f"best_{mode}_epoch").model_state_dict)
                fold_timestamps = [ts for ts in timestamps if key[0] <= ts <= key[1]]
                if 0 < len(fold_timestamps):
                    x = self.dataset.preprocess(dc=dc, timestamps=fold_timestamps)
                    x = x.to(device=device)
                    y_hats = self.model(x).cpu().numpy()
                    result.update({ts: y_hats[index] for index, ts in enumerate(fold_timestamps)})

        # move model back to cpu
        self.model.cpu()
        
        # set missed timestamps to array of np.nan values
        nan_y_hat = np.zeros((self.model.dim_output_feature)) * np.nan
        missing_timestamps = [ts for ts in timestamps if ts not in result]
        result.update({ts: nan_y_hat for ts in missing_timestamps})
        
        return result


    def predict(self, dc: DataCollection, timestamps: List[int], mode="val") -> Dict[float, np.array]:
        # load from y_hats_dict
        output = {ts: self.y_hats_dict[ts] for ts in timestamps if ts in self.y_hats_dict}
        output = {ts: y_hat for ts, y_hat in output.items() if not np.isnan(y_hat).max()}
        missing_timestamps = [ts for ts in timestamps if ts not in output]

        # predict missing timestamps and update self.y_hats_dict
        if 0 < len(missing_timestamps):
            output_missing_timestamp = self._predict(dc=dc, timestamps=timestamps, mode=mode)
            output.update(output_missing_timestamp)
            self.y_hats_dict.update(output)
            self.__rewrite_y_hats_dict = True

        return {k: v for k, v in sorted(output.items(), key=lambda item: item[0])}
    
    def calculate_metrics(
        self,
        dc: DataCollection,
        start_date: Union[str, datetime],
        stop_date: Union[str, datetime],
        mode: str = "val"
    ) -> Metrics:
         # prepare y and y_hat values for metrics
        stop_timestamp = self.dataset.label_transform.get_stop_timestamp(dc=dc)
        timestamps = self.get_timestamps(dc=dc, start_date=start_date, stop_date=stop_date)
        timestamps = [ts for ts in timestamps if ts < stop_timestamp]

        sf_generator = self.dataset.label_transform.transform_sf(dc=dc, timestamps=timestamps)
        y_array = np.concatenate([sf for sf in sf_generator])
        y_hat_dict = self.predict(dc=dc, timestamps=timestamps, mode=mode)   
        
        # convert y and y_hat values to torch.Tensor
        y = torch.from_numpy(y_array)
        y_hat = torch.from_numpy(np.array(list(y_hat_dict.values())))
        
        # create metrics
        metrics = Metrics(criterion=self.trainer.criterion, y=y, y_hat=y_hat)
        
        return metrics
 

    def draw_ohlc_plot(
            self,
            dc: DataCollection,
            start_date: Union[str, datetime],
            stop_date: Union[str, datetime],
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
        draw_predictions_based_on_labels(ohlc_ax=ohlc_ax, df=df)
        df.set_index("timestamp", inplace=True)

        # add legend and x tick angles
        ohlc_ax.legend()
        ohlc_ax.set_title("{} (from {} to {})".format(str(self), start_date, stop_date))
        for tick in ohlc_ax.get_xticklabels():
            tick.set_rotation(0)

        return fig, ohlc_ax, df

    def show_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str, mode: str = "val") -> None:
        _, _, _ = self.draw_ohlc_plot(dc=dc, start_date=start_date, stop_date=stop_date, mode=mode)
        plt.show()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return "{} {}".format(str(self.dataset), self.__model_type.__name__)
