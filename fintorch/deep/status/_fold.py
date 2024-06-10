from datetime import datetime
from typing import Dict, List

import numpy as np

from . import Epoch


class Fold:
    def __init__(self, index: int, train_timestamps: List[float], val_timestamps: List[float], test_timestamps: List[float]):
        self.__index = index
        self.__train_timestamps: List[float] = train_timestamps
        self.__val_timestamps: List[float] = val_timestamps
        self.__test_timestamp: List[float] = test_timestamps

        self.epochs_count: int = 0
        self.__epochs: List[Epoch] = []

    @property
    def index(self) -> int:
        return self.__index

    @property
    def epochs(self) -> List[Epoch]:
        return self.__epochs

    @property
    def last_epoch(self) -> Epoch:
        return self.epochs[-1]

    @property
    def completed(self) -> bool:
        return all(epoch.completed for epoch in self.epochs)

    @property
    def elapsed_time(self) -> float:
        elapsed_time = 0
        for epoch in self.epochs:
            if epoch.elapsed_time is not None:
                elapsed_time += epoch.elapsed_time

        return elapsed_time

    @property
    def total_time(self) -> float:
        return sum(epoch.total_time for epoch in self.epochs) * (self.epochs_count / len(self.epochs))

    @property
    def remaining_time(self) -> float:
        return self.total_time - self.elapsed_time

    @property
    def start_date(self) -> datetime:
        return self.train_start_datetime

    @property
    def stop_date(self) -> datetime:
        return self.test_stop_datetime

    @property
    def train_timestamps(self) -> List[float]:
        return self.__train_timestamps

    @property
    def val_timestamps(self) -> List[float]:
        return self.__val_timestamps

    @property
    def test_timestamps(self) -> List[float]:
        return self.__test_timestamp

    @property
    def timestamps(self) -> List[float]:
        return self.train_timestamps + self.val_timestamps + self.test_timestamps

    @property
    def train_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.train_timestamps[0])

    @property
    def val_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.val_timestamps[0])

    @property
    def test_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[0])

    @property
    def test_start_timestamp(self) -> float:
        return self.test_timestamps[0]

    @property
    def test_stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[-1])

    @property
    def test_stop_timestamp(self) -> float:
        return self.test_timestamps[-1]

    @property
    def best_train_epoch(self) -> Epoch:
        return self.__get_best_epoch(mode="train")

    @property
    def best_val_epoch(self) -> Epoch:
        return self.__get_best_epoch(mode="val")

    @property
    def best_test_epoch(self) -> Epoch:
        return self.__get_best_epoch(mode="test")

    @property
    def best_model_state_dict(self) -> Dict:
        return self.best_val_epoch.model_state_dict

    def __get_best_epoch(self, mode: str) -> Epoch:
        best_epoch = None
        for epoch in self.epochs:
            if best_epoch is None or best_epoch.compare(epoch, mode=mode, metric="loss") and epoch.completed:
                best_epoch = epoch

        return best_epoch

    def __eq__(self, other):
        return self.test_timestamps == other.test_timestamp

    def __str__(self):
        elapsed_time = self.last_epoch.elapsed_time
        remaining_time = self.last_epoch.remaining_time if not np.isnan(self.last_epoch.remaining_time) else 0
        total_time = self.last_epoch.total_time if not np.isnan(self.last_epoch.total_time) else 0

        elapsed_time_str = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
        remaining_time_str = datetime.strftime(datetime.utcfromtimestamp(remaining_time), '%H:%M:%S')
        total_time_str = datetime.strftime(datetime.utcfromtimestamp(total_time), '%H:%M:%S')

        return (
            (
                "Epoch ({}/{:<2}) ({} {} {})\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}\n"
                "{:<6} {:^10} {:^10} {:^10} {:^10} {:^10}"
            )
            .format(
                self.last_epoch.index, self.epochs_count, elapsed_time_str, remaining_time_str, total_time_str,
                "", "Last obj", "Best obj", "Last acc", "Best acc", "Best idx",

                "Train", self.last_epoch.train_loss, self.best_train_epoch.train_loss, self.last_epoch.train_accuracy,
                self.best_train_epoch.train_accuracy, self.best_train_epoch.index,

                "Val", self.last_epoch.val_loss, self.best_val_epoch.val_loss, self.last_epoch.val_accuracy,
                self.best_val_epoch.val_accuracy, self.best_val_epoch.index,

                "Test", self.last_epoch.test_loss, self.best_test_epoch.test_loss, self.last_epoch.test_accuracy,
                self.best_test_epoch.test_accuracy, self.best_test_epoch.index,

                "ToV", "-", self.best_train_epoch.val_loss, "-", self.best_train_epoch.val_accuracy,
                self.best_train_epoch.index,

                "VoT", "-", self.best_val_epoch.test_loss, "-", self.best_val_epoch.test_accuracy,
                self.best_val_epoch.index)
        )
