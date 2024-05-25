from abc import ABC
from typing import List

from torch import nn

from .._model import Model
from ....utils.hash import static_list_hash


class SFModel(Model, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module,

            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,

            num_hidden_layers: int,
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn
        )

        self.__dim_input_sequence: int = dim_input_sequence
        self.__dim_input_feature: int = dim_input_feature
        self.__dim_output_sequence: int = dim_output_sequence
        self.__dim_output_feature: int = dim_output_feature

        self.__num_hidden_layers: int = num_hidden_layers

        self.__static_hash: int = static_list_hash([
            super().static_hash,

            self.dim_input_sequence,
            self.dim_input_feature,
            self.dim_output_sequence,
            self.dim_output_feature,

            self.num_hidden_layers,
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
    def static_hash(self) -> int:
        return self.__static_hash

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
