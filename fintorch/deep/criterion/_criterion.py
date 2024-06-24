from abc import ABC, abstractmethod

import torch
from torch import nn

from fintorch.utils.hash import static_hash


class Criterion(nn.Module, ABC):
    def __init__(self, name: str, label_smoothing: float, reduction: str, classification_criterion: bool):
        super().__init__()
        self.__name: str = name
        self.__label_smoothing: float = label_smoothing
        self.__reduction: str = reduction
        self.__classification_criterion: bool = classification_criterion

        self.__static_hash: int = static_hash(self.name)

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def label_smoothing(self) -> str:
        return self.__label_smoothing

    @property
    def reduction(self) -> str:
        return self.__reduction

    @property
    def classification_criterion(self) -> bool:
        return self.__classification_criterion

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if torch.isnan(input).max() or torch.isinf(input).max():
            raise RuntimeError("NaN or Inf values found in the input of the criterion.")
        if torch.isnan(target).max() or torch.isinf(target).max():
            raise RuntimeError("NaN or Inf values found in the target of the criterion.")
        if torch.isnan(weight).max() or torch.isinf(weight).max():
            raise RuntimeError("NaN or Inf values found in the weight of the criterion.")

        output = self._forward(input=input, target=target, weight=weight)

        if torch.isnan(output).max() or torch.isinf(output).max():
            raise RuntimeError("NaN or Inf values found in the output of the criterion.")

        return output

    @abstractmethod
    def _forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def to_str(self, value: float) -> str:
        raise NotImplemented()

    @abstractmethod
    def compare(self, first: float, second: float) -> bool:
        raise NotImplemented()
