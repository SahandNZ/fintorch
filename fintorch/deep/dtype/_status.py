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
    def completed_folds(self) -> List[Fold]:
        return [fold for fold in self.folds if fold.completed]

    @property
    def fold_elapsed_times(self) -> List[float]:
        return self.__fold_elapsed_times

    @property
    def last_fold(self) -> Fold:
        return self.__folds[-1]

    @property
    def completed_folds_count(self) -> int:
        return len(self.completed_folds)

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

    def append_fold(self, fold: Fold) -> None:
        self.folds.append(fold)
        self.fold_elapsed_times.append(0)

    def update_elapsed_time(self, elapsed_time: float) -> None:
        self.fold_elapsed_times[-1] = elapsed_time

    def __str__(self):
        elapsed_time = datetime.strftime(datetime.utcfromtimestamp(self.elapsed_time), '%H:%M:%S')
        remaining_time = datetime.strftime(datetime.utcfromtimestamp(self.remaining_time), '%H:%M:%S')
        total_time = datetime.strftime(datetime.utcfromtimestamp(self.total_time), '%H:%M:%S')

        overall_vot_losses = [f.best_val_epoch.test_loss for f in self.folds]
        overall_vot_accuracies = [f.best_val_epoch.test_accuracy for f in self.folds]

        overall_vot_losses = [item for item in overall_vot_losses if not np.isnan(item)]
        overall_vot_accuracies = [item for item in overall_vot_accuracies if not np.isnan(item)]
        overall_vot_loss = np.round(np.array(overall_vot_losses).mean(), 4)
        overall_vot_accuracy = np.round(np.array(overall_vot_accuracies).mean(), 2)

        return (
            (
                "Fold ({}/{}) Epoch({}/{}) ({} {} {})\n\n"
                "{:<6} | {:^14} | {:^20} | {:^20}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
                "{:<6} | {:^6} {:^7} | {:^6} {:^7} {:^5} | {:^6} {:^7} {:^5}\n"
            )
            .format(
                self.last_fold.index, self.total_folds_count,
                self.last_fold.last_epoch.index, self.last_fold.epochs_count,
                elapsed_time, remaining_time, total_time,

                "", "Last", "Best", "Overall",
                "", "obj", "acc", "obj", "acc", "idx", "obj", "acc", "idx",

                "Train",
                # last epoch of last fold
                self.last_fold.last_epoch.train_loss,
                self.last_fold.last_epoch.train_accuracy,
                # best epoch of last fold
                self.last_fold.best_train_epoch.train_loss,
                self.last_fold.best_train_epoch.train_accuracy,
                self.last_fold.best_train_epoch.index,
                # overall of folds
                np.round(np.array([f.best_train_epoch.train_loss for f in self.folds]).mean(), 4),
                np.round(np.array([f.best_train_epoch.train_accuracy for f in self.folds]).mean(), 2),
                int(np.round(np.array([f.best_train_epoch.index for f in self.folds]).mean())),

                "Val",
                # last epoch of last fold
                self.last_fold.last_epoch.val_loss,
                self.last_fold.last_epoch.val_accuracy,
                # best epoch of last fold
                self.last_fold.best_val_epoch.val_loss,
                self.last_fold.best_val_epoch.val_accuracy,
                self.last_fold.best_val_epoch.index,
                # overall of folds
                np.round(np.array([f.best_val_epoch.val_loss for f in self.folds]).mean(), 4),
                np.round(np.array([f.best_val_epoch.val_accuracy for f in self.folds]).mean(), 2),
                int(np.round(np.array([f.best_val_epoch.index for f in self.folds]).mean())),

                "Test",
                # last epoch of last fold
                self.last_fold.last_epoch.test_loss,
                self.last_fold.last_epoch.test_accuracy,
                # best epoch of last fold
                self.last_fold.best_test_epoch.test_loss,
                self.last_fold.best_test_epoch.test_accuracy,
                self.last_fold.best_test_epoch.index,
                # overall of folds
                np.round(np.array([f.best_test_epoch.test_loss for f in self.folds]).mean(), 4),
                np.round(np.array([f.best_test_epoch.test_accuracy for f in self.folds]).mean(), 2),
                int(np.round(np.array([f.best_test_epoch.index for f in self.folds]).mean())),

                "ToV",
                # last epoch of last fold
                "-",
                "-",
                # best epoch of last fold
                self.last_fold.best_train_epoch.val_loss,
                self.last_fold.best_train_epoch.val_accuracy,
                "-",
                # overall of folds
                np.round(np.array([f.best_train_epoch.val_loss for f in self.folds]).mean(), 4),
                np.round(np.array([f.best_train_epoch.val_accuracy for f in self.folds]).mean(), 2),
                "-",

                "VoT",
                # last epoch of last fold
                "-",
                "-",
                # best epoch of last fold
                self.last_fold.best_val_epoch.test_loss,
                self.last_fold.best_val_epoch.test_accuracy,
                "-",
                # overall of folds
                overall_vot_loss,
                overall_vot_accuracy,
                "-",
            )
        )
