from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from ...component import Component
from ...utils.hash import static_list_hash


class Model(nn.Module, Component, ABC):
    def __init__(self, name: str, short_name: str, dim_sequence: int, dim_feature: int, dim_output: int,
                 num_hidden_layers: int, batch_norm: bool, dropout: float, activation_fn: nn.Module):
        nn.Module.__init__(self)
        Component.__init__(self, name=name, short_name=short_name, description="")

        self.__dim_sequence: int = dim_sequence
        self.__dim_feature: int = dim_feature
        self.__dim_output: int = dim_output
        self.__num_hidden_layers: int = num_hidden_layers
        self.__batch_norm: bool = batch_norm
        self.__dropout: float = dropout
        self.__activation_fn: nn.Module = activation_fn

        self.__device: torch.device = torch.device("cpu")
        self.__static_hash: int = static_list_hash([
            self.name,
            self.dim_sequence,
            self.dim_feature,
            self.dim_output,
            self.num_hidden_layers,
            self.batch_norm,
            self.dropout,
        ])

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
    def num_hidden_layers(self) -> int:
        return self.__num_hidden_layers

    @property
    def batch_norm(self) -> bool:
        return self.__batch_norm

    @property
    def dropout(self) -> float:
        return self.__dropout

    @property
    def activation_fn(self) -> nn.Module:
        return self.__activation_fn

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to(self.device)
        return self._forward(x)

    def reset(self, layers: List[nn.Module] = None):
        if layers is None:
            layers = list(self.children())

        for layer in layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif hasattr(layer, "children"):
                layer_children = list(layer.children())
                if 0 < len(layer_children):
                    self.reset(layers=layer_children)

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
