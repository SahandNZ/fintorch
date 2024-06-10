import gc
import math
from abc import ABC
from typing import Dict, Type, Union

import torch

from fintorch.deep.model import Model
from fintorch.utils.hash import static_list_hash


class Optimizer(ABC):
    def __init__(self, torch_optimizer_type: Type[torch.optim.Optimizer], **kwargs):
        self.__torch_optimizer_type: Type[torch.optim.Optimizer] = torch_optimizer_type
        self.__kwargs: Dict = kwargs

        # static hash calculations
        sorted_kwargs = {k: v for k, v in sorted(self.__kwargs.items(), key=lambda item: item[0])}
        lr = abs(int(math.log10(sorted_kwargs.pop("lr"))))
        weight_decay = abs(int(math.log10(sorted_kwargs.pop("weight_decay"))))
        self.__static_hash: int = static_list_hash(
            [
                self.__torch_optimizer_type.__name__,
                lr,
                weight_decay,
                *sorted_kwargs.values()
            ]
        )

    @property
    def torch_optimizer(self) -> torch.optim.Optimizer:
        return getattr(self, "__torch_optimizer")

    @property
    def lr(self) -> float:
        return next(iter(self.torch_optimizer.param_groups))['lr']

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def reset(self, model: Model):
        torch_optimizer = self.__torch_optimizer_type(params=model.parameters(), **self.__kwargs)
        setattr(self, "__torch_optimizer", torch_optimizer)

    def step(self):
        self.torch_optimizer.step()

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        delattr(self, "__torch_optimizer")
