from abc import ABC, abstractmethod

import torch
from torch import nn


class Criterion(nn.Module, ABC):
    def __init__(self, name: str, reduction: str, classification_criterion: bool):
        super().__init__()
        self.__name: str = name
        self.__reduction: str = reduction
        self.__classification_criterion: bool = classification_criterion

    @property
    def name(self) -> str:
        return self.__name

    @property
    def reduction(self) -> str:
        return self.__reduction

    @property
    def classification_criterion(self) -> bool:
        return self.__classification_criterion

    @abstractmethod
    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def to_str(self, value: float) -> str:
        raise NotImplemented()

    @abstractmethod
    def less_than(self, first: float, second: float) -> bool:
        raise NotImplemented()
