from typing import Dict, List, Union

import numpy as np

from ..criterion import Criterion


class Epoch:
    def __init__(self, index: int, criterion: Criterion) -> None:
        self.__index: int = index
        self.__criterion: Criterion = criterion

        self.train_batch_count: int = 0
        self.train_batch_times: List[float] = []
        self.train_batch_losses: List[float] = []
        self.train_batch_accuracies: List[int] = []

        self.val_batch_count: int = 0
        self.val_batch_times: List[float] = []
        self.val_batch_losses: List[float] = []
        self.val_batch_accuracies: List[int] = []

        self.test_batch_count: int = 0
        self.test_batch_times: List[float] = []
        self.test_batch_losses: List[float] = []
        self.test_batch_accuracies: List[int] = []

        self.model_state_dict: Union[Dict, None] = None
        self.completed: bool = False

    @property
    def index(self) -> int:
        return self.__index

    @property
    def criterion(self) -> Criterion:
        return self.__criterion

    @property
    def train_elapsed_time(self) -> float:
        return np.round(np.array(self.train_batch_times).sum(), 2)

    @property
    def train_total_time(self) -> float:
        return np.round(np.array(self.train_batch_times).mean() * self.train_batch_count, 2)

    @property
    def train_remaining_time(self) -> float:
        return np.round(self.train_total_time - self.train_elapsed_time, 2)

    @property
    def train_loss(self) -> float:
        return np.round(np.array(self.train_batch_losses).mean(), 4)

    @property
    def train_accuracy(self) -> float:
        return np.round(np.array(self.train_batch_accuracies).mean(), 2)

    @property
    def val_elapsed_time(self) -> float:
        return np.round(np.array(self.val_batch_times).sum(), 2)

    @property
    def val_total_time(self) -> float:
        return np.round(np.array(self.val_batch_times).mean() * self.val_batch_count, 2)

    @property
    def val_remaining_time(self) -> float:
        return np.round(self.val_total_time - self.val_elapsed_time, 2)

    @property
    def val_loss(self) -> float:
        return np.round(np.array(self.val_batch_losses).mean(), 4)

    @property
    def val_accuracy(self) -> float:
        return np.round(np.array(self.val_batch_accuracies).mean(), 2)

    @property
    def test_elapsed_time(self) -> float:
        return np.round(np.array(self.test_batch_times).sum(), 2)

    @property
    def test_total_time(self) -> float:
        return np.round(np.array(self.test_batch_times).mean() * self.test_batch_count, 2)

    @property
    def test_remaining_time(self) -> float:
        return np.round(self.test_total_time - self.test_elapsed_time, 2)

    @property
    def test_loss(self) -> float:
        return np.round(np.array(self.test_batch_losses).mean(), 4)

    @property
    def test_accuracy(self) -> float:
        return np.round(np.array(self.test_batch_accuracies).mean(), 2)

    @property
    def elapsed_time(self) -> float:
        return np.round(self.train_elapsed_time + self.val_elapsed_time + self.test_elapsed_time, 2)

    @property
    def total_time(self) -> float:
        return np.round(self.train_total_time + self.val_total_time + self.test_total_time, 2)

    @property
    def remaining_time(self) -> float:
        return np.round(self.total_time - self.elapsed_time, 2)

    def compare(self, other, mode: str, metric: str) -> bool:
        self_metric = getattr(self, f"{mode}_{metric}")
        other_metric = getattr(other, f"{mode}_{metric}")
        return other_metric <= self_metric if "loss" == metric else self_metric <= other_metric
