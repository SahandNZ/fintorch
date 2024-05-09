import math
from typing import List

import numpy as np
import torch
from torch import nn

from ._model import Model


class FeedForward(Model):
    def __init__(
            self,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,
            num_hidden_layers: int,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module
    ):
        super().__init__(
            name="Feed Forward",
            short_name="FF",
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn
        )

        input_layer = self.dim_input_sequence * self.dim_input_feature
        output_layer = self.dim_output_sequence * self.dim_output_feature
        powers = np.linspace(math.log2(input_layer), math.log2(output_layer), self.num_hidden_layers + 2)
        self.__layers = [input_layer] + [2 ** round(p) for p in powers[1:-1]] + [output_layer]
        self.__layers = [input_layer] * self.num_hidden_layers + [output_layer]

        modules = [nn.Flatten()]
        for index in range(len(self.layers) - 2):
            modules.append(nn.Linear(self.layers[index], self.layers[index + 1]))
            if self.batch_norm:
                modules.append(nn.BatchNorm1d(self.layers[index + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(self.layers[-2], self.layers[-1]))
        if self.batch_norm:
            modules.append(nn.BatchNorm1d(self.layers[-1]))

        self.net = nn.Sequential(*modules)

    @property
    def layers(self) -> List[int]:
        return self.__layers

    def _forward(self, x: torch.tensor):
        y_hat = self.net(x)
        y_hat = y_hat.reshape(-1, self.dim_output_sequence, self.dim_output_feature)
        if self.activation_fn is not None:
            y_hat = self.activation_fn(y_hat)

        return y_hat
