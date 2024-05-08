import torch

from ._criterion import Criterion


class FocalMSE(Criterion):
    def __init__(self):
        super().__init__(name="FocalMSE", reduction='mean', classification_criterion=True)

    def _forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = (target - input) ** 2

    def to_str(self, value: float) -> str:
        return "Focal MSE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
