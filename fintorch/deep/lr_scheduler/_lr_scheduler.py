from typing import Dict, Type

import torch
from ..optimizer import Optimizer


class LRScheduler:
    def __init__(self, lr_scheduler: Type[torch.optim.lr_scheduler.LRScheduler], **kwargs):
        self.__torch_lr_scheduler_class: Type[torch.optim.lr_scheduler.LRScheduler] = lr_scheduler
        self.__kwargs: Dict = kwargs

        self.__core: torch.optim.lr_scheduler.LRScheduler = None

    @property
    def core(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.__core

    def reset(self, optimizer: Optimizer):
        self.__core = self.__torch_lr_scheduler_class(optimizer=optimizer.core, **self.__kwargs)

    def step(self):
        self.core.step()
