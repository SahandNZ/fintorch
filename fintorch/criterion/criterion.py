from abc import ABC, abstractmethod

import torch
from torch import nn


class Criterion(nn.Module, ABC):
    def __init__(self, reduction: str):
        super().__init__()
        self.__reduction: str = reduction

    @property
    def reduction(self) -> str:
        return self.__reduction

    @abstractmethod
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def to_str(self, value: float) -> str:
        raise NotImplemented()

    @abstractmethod
    def less_than(self, first: float, second: float) -> bool:
        raise NotImplemented()
