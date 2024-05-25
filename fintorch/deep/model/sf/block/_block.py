from abc import ABC

from torch import nn


class Block(nn.Module, ABC):
    def __init__(self):
        super().__init__()
