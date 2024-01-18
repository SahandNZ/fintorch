import math
from typing import List

import numpy as np
import torch
from torch import nn

from ._model import Model


class FeedForward(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, dropout: float = 0,
                 activation_fn: nn.Module = None, num_hidden_layers: int = 4, batch_norm: bool = False):
        super().__init__(
            name="Feed Froward",
            short_name="FF",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            dropout=dropout,
            activation_fn=activation_fn
        )

        self.__num_hidden_layers: int = num_hidden_layers

        input_layer = self.dim_sequence * self.dim_feature
        output_layer = self.dim_output
        powers = np.linspace(math.log2(input_layer), math.log2(output_layer), self.num_hidden_layers + 2)
        self.__layers = [2 ** round(p) for p in powers]

        modules = [nn.Flatten()]
        for index in range(len(self.layers) - 2):
            modules.append(nn.Linear(self.layers[index], self.layers[index + 1]))
            if batch_norm:
                modules.append(nn.BatchNorm1d(self.layers[index + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(self.layers[-2], self.layers[-1]))
        if batch_norm:
            modules.append(nn.BatchNorm1d(self.layers[-1]))
        if activation_fn is not None:
            modules.append(activation_fn)

        self.net = nn.Sequential(*modules)

    @property
    def num_hidden_layers(self) -> int:
        return self.__num_hidden_layers

    @property
    def layers(self) -> List[int]:
        return self.__layers

    def forward(self, x: torch.tensor):
        return self.net(x)
