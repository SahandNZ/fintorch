from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from fintorch.component import Component
from fintorch.utils.hash import static_list_hash


class Model(nn.Module, Component, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,
            num_hidden_layers: int,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module
    ):
        nn.Module.__init__(self)
        Component.__init__(self, name=name, short_name=short_name, description="")

        self.__dim_input_sequence: int = dim_input_sequence
        self.__dim_input_feature: int = dim_input_feature
        self.__dim_output_sequence: int = dim_output_sequence
        self.__dim_output_feature: int = dim_output_feature
        self.__num_hidden_layers: int = num_hidden_layers
        self.__dropout: float = dropout
        self.__batch_norm: bool = batch_norm
        self.__activation_fn: nn.Module = activation_fn

        self.__device: torch.device = torch.device("cpu")
        self.__static_hash: int = static_list_hash([
            self.name,
            self.dim_input_sequence,
            self.dim_input_feature,
            self.dim_output_sequence,
            self.dim_output_feature,
            self.num_hidden_layers,
            int(self.dropout * 10 ** 2),
            self.batch_norm,
        ])

    @property
    def dim_input_sequence(self) -> int:
        return self.__dim_input_sequence

    @property
    def dim_input_feature(self) -> int:
        return self.__dim_input_feature

    @property
    def dim_output_sequence(self) -> int:
        return self.__dim_output_sequence

    @property
    def dim_output_feature(self) -> int:
        return self.__dim_output_feature

    @property
    def num_hidden_layers(self) -> int:
        return self.__num_hidden_layers

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
    def static_hash(self) -> int:
        return self.__static_hash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).max() or torch.isinf(x).max():
            raise RuntimeError("NaN or Inf values found in the input of the model.")

        output = self._forward(x)
        if torch.isnan(output).max() or torch.isinf(output).max():
            raise RuntimeError("NaN or Inf values found in the output of the model.")

        return output

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
