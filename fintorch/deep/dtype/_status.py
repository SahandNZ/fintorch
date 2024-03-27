from datetime import datetime
from typing import List

import numpy as np

from ._fold import Fold


class Status:
    def __init__(self, total_folds_count: int):
        self.__total_folds_count = total_folds_count

        self.__folds: List[Fold] = []
        self.__fold_elapsed_times: List[float] = []

    @property
    def total_folds_count(self) -> int:
        return self.__total_folds_count

    @property
    def folds(self) -> List[Fold]:
        return self.__folds

    @property
    def fold_elapsed_times(self) -> List[float]:
        return self.__fold_elapsed_times

    @property
    def last_fold(self) -> Fold:
        return self.__folds[-1]

    @property
    def completed_folds_count(self) -> int:
        return sum(1 for fold in self.folds if fold.completed)

    @property
    def remaining_folds_count(self) -> int:
        return self.total_folds_count - self.completed_folds_count

    @property
    def elapsed_time(self) -> float:
        return sum(fold_time for fold, fold_time in zip(self.folds, self.fold_elapsed_times) if fold.completed)

    @property
    def total_time(self) -> float:
        if 1 < self.completed_folds_count:
            return self.elapsed_time / self.completed_folds_count * self.total_folds_count
        else:
            return 0

    @property
    def remaining_time(self) -> float:
        return self.total_time - self.elapsed_time

    @property
    def overall_vot_accuracy(self) -> float:
        accuracy_values = [fold.best_val_epoch.test_accuracy for fold in self.folds]
        return sum(accuracy_values) / len(accuracy_values)

    @property
    def overall_vot_loss(self) -> float:
        loss_values = [fold.best_val_epoch.test_loss for fold in self.folds]
        valid_loss_values = [value for value in loss_values if not np.isinf(value) and not np.isinf(value)]
        return sum(valid_loss_values) / len(valid_loss_values) if 0 < len(valid_loss_values) else 0

    @property
    def overall_vot_str(self) -> str:
        return "Overall VoT  {:<6.4f}  {:<5.1f}%".format(self.overall_vot_loss, self.overall_vot_accuracy)

    def append_fold(self, fold: Fold) -> None:
        self.folds.append(fold)
        self.fold_elapsed_times.append(0)

    def update_elapsed_time(self, elapsed_time: float) -> None:
        self.fold_elapsed_times[-1] = elapsed_time

    def __str__(self):
        elapsed_time = datetime.strftime(datetime.utcfromtimestamp(self.elapsed_time), '%H:%M:%S')
        remaining_time = datetime.strftime(datetime.utcfromtimestamp(self.remaining_time), '%H:%M:%S')
        total_time = datetime.strftime(datetime.utcfromtimestamp(self.total_time), '%H:%M:%S')

        return "Fold ({}/{}) ({} {} {})\n{}\n{}" \
            .format(
            self.completed_folds_count,
            self.total_folds_count,
            elapsed_time,
            remaining_time,
            total_time,
            self.overall_vot_str,
            str(self.last_fold)
        )
