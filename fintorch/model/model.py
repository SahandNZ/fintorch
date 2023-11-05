import os
from abc import abstractmethod, ABC

import torch
from torch import nn

from fintorch.component import Component


class Model(nn.Module, Component, ABC):
    def __init__(self, name: str, short_name: str, auto_cuda: bool):
        nn.Module.__init__(self)
        Component.__init__(self, name=name, short_name=short_name, description="")
        self.__auto_cuda: bool = auto_cuda

    @property
    def auto_cuda(self):
        return self.__auto_cuda

    @auto_cuda.setter
    def auto_cuda(self, value: bool):
        self.__auto_cuda: bool = value

    @property
    def device(self) -> str:
        return torch.device('cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError()

    def reset(self, xavier: bool = True):
        for layer in self.children():
            if xavier and (type(layer) == nn.Linear or type(layer) == nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def load(self, path: str):
        self.load_state_dict(torch.load(os.path.join(path), map_location=self.device))
        self.eval()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def __call__(self, *args):
        self.to(self.device)
        return self.forward(*args)
