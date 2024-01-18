from typing import List

import numpy as np
import torch
import math
from torch import nn

from ._model import Model
from .block import Residual1D


class ResNet1D(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, dropout: float = 0,
                 activation_fn: nn.Module = None, num_hidden_blocks: int = 4, batch_norm: bool = False):
        super().__init__(
            name="Residual Network 1D",
            short_name="RN-1D",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            dropout=dropout,
            activation_fn=activation_fn
        )
        self.__num_hidden_blocks: int = num_hidden_blocks

        input_block = self.dim_sequence * self.dim_feature
        output_block = self.dim_output
        powers = np.linspace(math.log2(input_block), math.log2(output_block), self.num_hidden_blocks + 2)
        self.__blocks = [2 ** round(p) for p in powers]

        modules = [nn.Flatten()]
        for index in range(len(self.blocks) - 2):
            block = Residual1D(input_dim=self.blocks[index], output_dim=self.blocks[index + 1], batch_norm=batch_norm)
            modules.append(block)
            modules.append(nn.Dropout(dropout))
            modules.append(nn.LeakyReLU())

        modules.append(Residual1D(input_dim=self.blocks[-2], output_dim=self.blocks[-1], batch_norm=batch_norm))
        if activation_fn is not None:
            modules.append(activation_fn)

        self.net = nn.Sequential(*modules)

    @property
    def num_hidden_blocks(self) -> int:
        return self.__num_hidden_blocks

    @property
    def blocks(self) -> List[int]:
        return self.__blocks

    def forward(self, x: torch.tensor):
        return self.net(x)
