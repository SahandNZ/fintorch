from typing import Dict, Type

import torch

from .optimizer import Optimizer
from ..utils.hash import static_list_hash


class LrScheduler:
    def __init__(self, torch_lr_scheduler_type: Type[torch.optim.lr_scheduler.LRScheduler], **kwargs):
        self.__torch_lr_scheduler_type: Type[torch.optim.lr_scheduler.LRScheduler] = torch_lr_scheduler_type
        self.__kwargs: Dict = kwargs

        self.__static_hash: int = static_list_hash([self.__torch_lr_scheduler_type.__name__])

    @property
    def torch_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return getattr(self, "__torch_lr_scheduler")

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def reset(self, optimizer: Optimizer):
        torch_lr_scheduler = self.__torch_lr_scheduler_type(optimizer=optimizer.torch_optimizer, **self.__kwargs)
        setattr(self, "__torch_lr_scheduler", torch_lr_scheduler)

    def step(self):
        self.torch_lr_scheduler.step()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        delattr(self, "__torch_lr_scheduler")
