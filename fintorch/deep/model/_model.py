from abc import ABC, abstractmethod

import torch
from torch import nn

from ...component import Component


class Model(nn.Module, Component, ABC):
    def __init__(self, name: str, short_name: str, dim_sequence: int, dim_feature: int, dim_output: int, dropout: float,
                 activation_fn: nn.Module):
        nn.Module.__init__(self)
        Component.__init__(self, name=name, short_name=short_name, description="")

        self.__dim_sequence: int = dim_sequence
        self.__dim_feature: int = dim_feature
        self.__dim_output: int = dim_output
        self.__dropout: float = dropout
        self.__activation_fn: nn.Module = activation_fn

    @property
    def dim_sequence(self) -> int:
        return self.__dim_sequence

    @property
    def dim_feature(self) -> int:
        return self.__dim_feature

    @property
    def dim_output(self) -> int:
        return self.__dim_output

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def activation_fn(self) -> nn.Module:
        return self.__activation_fn

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self, xavier: bool = True):
        for layer in self.children():
            if xavier and (type(layer) == nn.Linear or type(layer) == nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
