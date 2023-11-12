from abc import ABC
from typing import Dict, Type

import torch

from fintorch.model.model import Model


class Optimizer(ABC):
    def __init__(self, cls: Type, **kwargs):
        self.__cls: Type = cls
        self.__kwargs: Dict = kwargs

        self.__torch_optimizer: torch.optim.Optimizer = None

    @property
    def torch_optimizer(self) -> torch.optim.Optimizer:
        return self.__torch_optimizer

    @property
    def lr(self) -> float:
        return next(iter(self.torch_optimizer.param_groups))['lr']

    def reset(self, model: Model):
        self.__torch_optimizer = self.__cls(params=model.parameters(), **self.__kwargs)

    def step(self):
        self.__torch_optimizer.step()

    def zero_grad(self):
        self.__torch_optimizer.zero_grad()
