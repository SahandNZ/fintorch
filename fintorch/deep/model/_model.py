from abc import abstractmethod
from typing import List

import torch
from torch import nn

from ...component import Component
from ...utils.hash import static_list_hash


class Model(nn.Module, Component):
    def __init__(
            self,
            name: str,
            short_name: str,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module
    ):
        nn.Module.__init__(self)
        Component.__init__(self, name=name, short_name=short_name, description="")

        self.__dropout: float = dropout
        self.__batch_norm: bool = batch_norm
        self.__activation_fn: nn.Module = activation_fn
        self.__device: torch.device = torch.device("cpu")

        self.__static_hash: int = static_list_hash([
            self.name,
            int(self.dropout * 10 ** 2),
            self.batch_norm,
        ])

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def batch_norm(self) -> bool:
        return self.__batch_norm

    @property
    def activation_fn(self) -> nn.Module:
        return self.__activation_fn

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @abstractmethod
    def static_hash(self) -> int:
        return self.__static_hash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).max() or torch.isinf(x).max():
            raise RuntimeError("NaN or Inf values found in the input of the model.")

        output = self._forward(x)
        if torch.isnan(output).max() or torch.isinf(output).max():
            raise RuntimeError("NaN or Inf values found in the output of the model.")

        return output

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, layers: List[nn.Module] = None):
        raise NotImplementedError()
