import torch

from fintorch.data_loader.data_loader import DataLoader


class UnderSamplingDataLoader(DataLoader):
    def __init__(self, batch_size: int, shuffle: bool = True, auto_cuda: bool = True, select_randomly: bool = False):
        super().__init__(batch_size, shuffle, auto_cuda)
        self.select_randomly: bool = select_randomly

    def under_sample(self):
        labels = torch.unique(self._dataset.y, dim=0)
        min_label_length = min([(self._dataset.y == cls).min(dim=1).values.sum() for cls in labels])
        x_dict, y_dict = {}, {}
        for label in labels:
            label_indices = (self._dataset.y == label).min(dim=1).values
            x = self._dataset.x[label_indices]
            y = self._dataset.y[label_indices]

            if len(y) == min_label_length:
                select_indices = torch.arange(len(y))
            elif self.select_randomly:
                select_indices = torch.randint(high=len(y), size=(min_label_length,))
            else:
                select_indices = torch.arange(len(y))[:min_label_length]

            x_dict[label] = x[select_indices]
            y_dict[label] = y[select_indices]

        x = torch.concat(tuple(x_dict.values()), dim=0)
        y = torch.concat(tuple(y_dict.values()), dim=0)
        self._dataset.preset(x=x, y=y, df=None)

    def __iter__(self):
        self.under_sample()
        super().__iter__()

        return self
