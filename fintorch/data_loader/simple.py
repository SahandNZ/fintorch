from typing import Tuple

import torch

from fintorch.data_loader.data_loader import DataLoader


class SimpleDataLoader(DataLoader):
    def __init__(self, batch_size: int, shuffle: bool = True, auto_cuda: bool = True):
        super().__init__(batch_size, shuffle, auto_cuda)

        self._index: int = None

    def __iter__(self):
        self._index = -1
        if self.shuffle:
            random_index = torch.randperm(len(self._dataset))
            self._dataset.x = self._dataset.x[random_index]
            self._dataset.y = self._dataset.y[random_index] if self._dataset.y is not None else None

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self._index * self.batch_size < len(self._dataset.x):
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size

            batch_x = self._dataset.x[start_index: stop_index]
            batch_y = self._dataset.y[start_index: stop_index] if self._dataset.y is not None else None

            return batch_x, batch_y

        else:
            raise StopIteration
