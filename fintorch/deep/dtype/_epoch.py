import math
from typing import Dict, List

from fintorch.deep.criterion import Criterion


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

        self.model_state_dict: Dict = None

    @property
    def index(self) -> int:
        return self.__index

    @property
    def criterion(self) -> Criterion:
        return self.__criterion

    @property
    def train_elapsed_time(self) -> float:
        return sum(self.train_batch_times)

    @property
    def train_total_time(self) -> float:
        if 0 < len(self.train_batch_times):
            return self.train_elapsed_time * (self.train_batch_count / len(self.train_batch_times))
        else:
            return 0

    @property
    def train_remaining_time(self) -> float:
        return self.train_total_time - self.train_elapsed_time

    @property
    def train_loss(self) -> float:
        # TODO Define worst case value in criterion to use it in such cases
        if 0 < len(self.train_batch_losses):
            return sum(self.train_batch_losses) / len(self.train_batch_losses)
        else:
            return math.inf

    @property
    def train_accuracy(self) -> float:
        if 0 < len(self.train_batch_accuracies):
            return sum(self.train_batch_accuracies) / len(self.train_batch_accuracies)
        else:
            return 0

    @property
    def val_elapsed_time(self) -> float:
        return sum(self.val_batch_times)

    @property
    def val_total_time(self) -> float:
        if 0 < len(self.val_batch_times):
            return self.val_elapsed_time * (self.val_batch_count / len(self.val_batch_times))
        else:
            return 0

    @property
    def val_remaining_time(self) -> float:
        return self.val_total_time - self.val_elapsed_time

    @property
    def val_loss(self) -> float:
        if 0 < len(self.val_batch_losses):
            return sum(self.val_batch_losses) / len(self.val_batch_losses)
        else:
            return math.inf

    @property
    def val_accuracy(self) -> float:
        if 0 < len(self.val_batch_accuracies):
            return sum(self.val_batch_accuracies) / len(self.val_batch_accuracies)
        else:
            return 0

    @property
    def test_elapsed_time(self) -> float:
        return sum(self.test_batch_times)

    @property
    def test_total_time(self) -> float:
        if 0 < len(self.test_batch_times):
            return self.test_elapsed_time * (self.test_batch_count / len(self.test_batch_times))
        else:
            return 0

    @property
    def test_remaining_time(self) -> float:
        return self.test_total_time - self.test_elapsed_time

    @property
    def test_loss(self) -> float:
        if 0 < len(self.test_batch_losses):
            return sum(self.test_batch_losses) / len(self.test_batch_losses)
        else:
            return math.inf

    @property
    def test_accuracy(self) -> float:
        if 0 < len(self.test_batch_accuracies):
            return sum(self.test_batch_accuracies) / len(self.test_batch_accuracies)
        else:
            return 0

    @property
    def elapsed_time(self) -> float:
        return self.train_elapsed_time + self.val_elapsed_time + self.test_elapsed_time

    @property
    def total_time(self) -> float:
        return self.train_total_time + self.val_total_time + self.test_total_time

    @property
    def remaining_time(self) -> float:
        return self.total_time - self.elapsed_time

    def compare(self, other, mode: str) -> bool:
        self_loss = getattr(self, f"{mode}_loss")
        other_loss = getattr(other, f"{mode}_loss")
        return self.criterion.compare(self_loss, other_loss)
