import math
import warnings

from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.cross_validation.fold import Fold


class SlidingWindowCrossValidation(CrossValidation):
    def __init__(self, window_size: int, train_percentage: float, dev_percentage: float, window_step: int = None):
        super().__init__(train_percentage, dev_percentage)
        self.__window_size: int = window_size
        self.__window_step: int = window_step

        self.__train_length: int = int(self.__window_size * self.train_percentage)
        self.__dev_length: int = int(self.__window_size * self.dev_percentage)
        self.__test_length: int = int(self.__window_size - self.__train_length - self.__dev_length)

        self._index: int = None
        self.__fold_count: int = None

    @property
    def window_size(self) -> int:
        return self.__window_size

    @property
    def window_step(self) -> int:
        if self.__window_step is None:
            return self.__test_length
        return self.__window_step

    @property
    def train_length(self) -> int:
        return self.__train_length

    @property
    def dev_length(self) -> int:
        return self.__dev_length

    @property
    def test_length(self) -> int:
        return self.__test_length

    @property
    def fold_count(self) -> int:
        if self.__fold_count is None:
            raise Exception("Dataset must be set before accessing to this property.")
        return self.__fold_count

    def __iter__(self):
        if len(self.dataset) < self.window_size:
            warnings.warn(f"Dataset should have at least {self.window_size} samples.")
            self.__window_size = len(self.dataset)

        self._index = -1
        self.__fold_count = math.ceil((len(self.dataset) - self.window_size) / self.window_step) + 1

        return self

    def __next__(self) -> Fold:
        self._index += 1
        if self.index < self.fold_count:
            train_start_index = self.index * self.window_step
            train_stop_index = train_start_index + self.train_length
            dev_stop_index = train_stop_index + self.dev_length
            test_stop_index = dev_stop_index + self.test_length

            train_set = self.dataset[train_start_index:train_stop_index]
            dev_set = self.dataset[train_stop_index:dev_stop_index]
            test_set = self.dataset[dev_stop_index:test_stop_index]

            fold = Fold(self.index, train_set, dev_set, test_set)

            return fold

        else:
            raise StopIteration()
