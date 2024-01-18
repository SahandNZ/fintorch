import math
from typing import Iterator

from ._cross_validation import CrossValidation
from ..dtype import Dataset, Fold


class SlidingWindowCrossValidation(CrossValidation):
    def __init__(self, train_percentage: float, dev_percentage: float, window_length: int):
        super().__init__(train_percentage=train_percentage, dev_percentage=dev_percentage)
        self.__window_length: int = window_length

        self._train_length = None
        self._dev_length = None
        self._test_length = None
        self._folds_count = None

    @property
    def window_length(self) -> int:
        return self.__window_length

    def __call__(self, dataset: Dataset) -> Iterator:
        super().__call__(dataset=dataset)

        self._train_length = int(self.window_length * self.train_percentage / 100)
        self._dev_length = int(self.window_length * self.dev_percentage / 100)
        self._test_length = int(self.window_length - self.train_length - self.dev_length)
        self._folds_count = math.ceil((self.samples_count - self.window_length) / self.test_length) + 1

        return self.__iter__()

    def __next__(self) -> Fold:
        self._index += 1
        if self.index < self.folds_count:
            start_timestamp = int(self.start_date.timestamp() + self.index * self.test_length * self.time_frame)
            fold = self._create_fold_by_start_timestamp(start_timestamp=start_timestamp)
            return fold

        else:
            raise StopIteration
