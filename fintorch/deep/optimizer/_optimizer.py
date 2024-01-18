from abc import ABC
from typing import Dict, Type

import torch
from fintorch.deep.model import Model


class Optimizer(ABC):
    def __init__(self, optimizer: Type[torch.optim.Optimizer], **kwargs):
        self.__torch_optimizer_class: Type[torch.optim.Optimizer] = optimizer
        self.__kwargs: Dict = kwargs

        self.__core: torch.optim.Optimizer = None

    @property
    def core(self) -> torch.optim.Optimizer:
        return self.__core

    @property
    def lr(self) -> float:
        return next(iter(self.core.param_groups))['lr']

    def reset(self, model: Model):
        self.__core = self.__torch_optimizer_class(params=model.parameters(), **self.__kwargs)

    def step(self):
        self.core.step()

    def zero_grad(self):
        self.core.zero_grad()
