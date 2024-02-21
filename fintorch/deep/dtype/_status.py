from datetime import datetime
from typing import List

from ._fold import Fold


class Status:
    def __init__(self, folds_count: int):
        self.__folds_count = folds_count

        self.__folds: List[Fold] = []
        self.__fold_elapsed_times: List[float] = []

    @property
    def folds_count(self) -> int:
        return self.__folds_count

    @property
    def completed_folds_count(self) -> int:
        return len(self.completed_folds)

    @property
    def folds(self) -> List[Fold]:
        return self.__folds

    @property
    def completed_folds(self) -> List[Fold]:
        return self.folds

    @property
    def last_fold(self) -> Fold:
        return self.__folds[-1]

    @property
    def fold_elapsed_times(self) -> List[float]:
        return self.__fold_elapsed_times

    @property
    def elapsed_time(self) -> float:
        return sum(fold_time for fold, fold_time in zip(self.folds, self.fold_elapsed_times) if fold.completed)

    @property
    def total_time(self) -> float:
        return self.elapsed_time * self.folds_count / self.completed_folds_count

    @property
    def remaining_time(self) -> float:
        return self.total_time - self.elapsed_time

    def append_fold(self, fold: Fold) -> None:
        self.folds.append(fold)
        self.fold_elapsed_times.append(0)

    def update_elapsed_time(self, elapsed_time: float) -> None:
        self.fold_elapsed_times[-1] = elapsed_time

    def __str__(self):
        elapsed_time = datetime.strftime(datetime.utcfromtimestamp(self.elapsed_time), '%H:%M:%S')
        remaining_time = datetime.strftime(datetime.utcfromtimestamp(self.remaining_time), '%H:%M:%S')
        total_time = datetime.strftime(datetime.utcfromtimestamp(self.total_time), '%H:%M:%S')

        return "Fold ({}/{}) ({} {} {})\n{}" \
            .format(self.completed_folds_count, self.folds_count, elapsed_time, remaining_time, total_time,
                    str(self.last_fold))
