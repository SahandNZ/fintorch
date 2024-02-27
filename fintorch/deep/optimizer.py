from abc import ABC
from typing import Dict, Type

import torch
from fintorch.deep.model import Model
from fintorch.utils.hash import static_list_hash


class Optimizer(ABC):
    def __init__(self, torch_optimizer_type: Type[torch.optim.Optimizer], **kwargs):
        self.__torch_optimizer_type: Type[torch.optim.Optimizer] = torch_optimizer_type
        self.__kwargs: Dict = kwargs

        self.__torch_optimizer: torch.optim.Optimizer = None

    @property
    def torch_optimizer(self) -> torch.optim.Optimizer:
        return self.__torch_optimizer

    @property
    def lr(self) -> float:
        return next(iter(self.torch_optimizer.param_groups))['lr']

    def reset(self, model: Model):
        self.__torch_optimizer = self.__torch_optimizer_type(params=model.parameters(), **self.__kwargs)

    def step(self):
        self.torch_optimizer.step()

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def __hash__(self):
        sorted_kwargs = {k: v for k, v in sorted(self.__kwargs.items(), key=lambda item: item[0])}
        lr = sorted_kwargs.pop("lr") / 1e-5
        weight_decay = sorted_kwargs.pop("weight_decay") / 1e-5
        return static_list_hash(
            [
                self.__torch_optimizer_type.__name__,
                lr,
                weight_decay,
                *sorted_kwargs.values()
            ]
        )
