import torch

from ._optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(optimizer=torch.optim.Adam, **kwargs)
