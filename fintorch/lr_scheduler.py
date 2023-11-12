from typing import Dict, Type

import torch

from fintorch.optimizer import Optimizer


class LRScheduler:
    def __init__(self, cls: Type, **kwargs):
        self.__cls: Type = cls
        self.__kwargs: Dict = kwargs

        self.__torch_scheduler: torch.optim.lr_scheduler.LRScheduler = None

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.__torch_scheduler

    def reset(self, optimizer: Optimizer):
        self.__torch_scheduler = self.__cls(optimizer=optimizer.torch_optimizer, **self.__kwargs)

    def step(self):
        self.scheduler.step()
