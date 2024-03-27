from typing import Dict, Type

import torch

from .optimizer import Optimizer
from ..utils.hash import static_list_hash


class LrScheduler:
    def __init__(self, torch_lr_scheduler_type: Type[torch.optim.lr_scheduler.LRScheduler], **kwargs):
        self.__torch_lr_scheduler_type: Type[torch.optim.lr_scheduler.LRScheduler] = torch_lr_scheduler_type
        self.__kwargs: Dict = kwargs

        self.__torch_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None

        self.__static_hash: int = static_list_hash([self.__torch_lr_scheduler_type.__name__])

    @property
    def torch_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.__torch_lr_scheduler

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def reset(self, optimizer: Optimizer):
        self.__torch_lr_scheduler = self.__torch_lr_scheduler_type(optimizer=optimizer.torch_optimizer, **self.__kwargs)

    def step(self):
        self.torch_lr_scheduler.step()
