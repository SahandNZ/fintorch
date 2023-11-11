from typing import List

import torch
from torch import nn

from fintorch.model.model import Model


class FeedForward(Model):
    def __init__(self, layers: List[int], dropout: float, activation_fn: nn.Module = None):
        super().__init__(name="Feed Froward", short_name="FF")
        self.layers: List[int] = layers

        modules = [nn.Flatten()]
        for index in range(len(self.layers) - 2):
            modules.append(nn.Linear(layers[index], layers[index + 1]))
            modules.append(nn.BatchNorm1d(layers[index + 1]))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(layers[-2], layers[-1]))
        modules.append(nn.BatchNorm1d(layers[-1]))
        if activation_fn is not None:
            modules.append(activation_fn)

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.tensor):
        return self.net(x)
