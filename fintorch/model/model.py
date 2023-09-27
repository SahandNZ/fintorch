import os
from abc import abstractmethod, ABC

import torch
from torch import nn


class Model(nn.Module, ABC):
    def __init__(self, auto_cuda: bool):
        super().__init__()
        self.__auto_cuda: bool = auto_cuda

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def auto_cuda(self):
        return self.__auto_cuda

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
        print(self.device)
        self.forward(args)
