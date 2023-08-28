from typing import List

import torch
from torch import nn

from fintorch.model.model import Model


class ResNet1D(Model):
    def __init__(self, block: nn.Module, layers: List[int], dropout: float, activation_fn: nn.Module = None,
                 auto_cuda: bool = True):
        super().__init__(auto_cuda)

        modules = [nn.Flatten()]
        for index in range(len(layers) - 2):
            modules.append(block(input_dim=layers[index], output_dim=layers[index + 1]))
            modules.append(nn.Dropout(dropout))
            modules.append(nn.ReLU())

        modules.append(block(input_dim=layers[-2], output_dim=layers[-1]))
        if activation_fn is not None:
            modules.append(activation_fn)

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.tensor):
        return self.net(x)
