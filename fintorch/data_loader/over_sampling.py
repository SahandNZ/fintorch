from typing import Tuple

import torch

from fintorch.data_loader.data_loader import DataLoader


class OverSamplingDataLoader(DataLoader):
    def __init__(self, batch_size: int, shuffle: bool = True, auto_cuda: bool = True, select_randomly: bool = False):
        super().__init__(batch_size, shuffle, auto_cuda)
        self.select_randomly: bool = select_randomly

        self._index: int = None

    def __iter__(self):
        self._index = -1

        # resampling part
        labels = torch.unique(self._dataset.y, dim=0)
        max_label_length = max([(self._dataset.y == cls).sum() for cls in labels])
        x_dict, y_dict = {}, {}
        for label in labels:
            label_indices = self._dataset.y == label
            x = self._dataset.x[label_indices]
            y = self._dataset.y[label_indices]

            if len(y) == max_label_length:
                select_indices = torch.arange(len(y))
            elif self.select_randomly:
                select_indices = torch.randint(high=len(y), size=(max_label_length,))
            else:
                indices = torch.arange(len(y))
                repeat_factor = max_label_length // len(y)
                repeat_remainder = max_label_length - len(y) * repeat_factor
                repeated_indices = tuple([indices] * repeat_factor + [indices[:repeat_remainder]])
                select_indices = torch.cat(repeated_indices)

            x_dict[label] = x[select_indices]
            y_dict[label] = y[select_indices]

        self._dataset.x = torch.concat(tuple(x_dict.values()), dim=0)
        self._dataset.y = torch.concat(tuple(y_dict.values()), dim=0)

        # shuffle part
        if self.shuffle:
            random_index = torch.randperm(len(self._dataset))
            self._dataset.x = self._dataset.x[random_index]
            self._dataset.y = self._dataset.y[random_index]

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._index += 1

        if self.index * self.batch_size < len(self._dataset.x):
            start_index = self.index * self.batch_size
            stop_index = start_index + self.batch_size

            batch_x = self._dataset.x[start_index: stop_index]
            batch_y = self._dataset.y[start_index: stop_index]

            return batch_x, batch_y

        else:
            raise StopIteration
