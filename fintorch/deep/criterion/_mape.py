import torch

from ._criterion import Criterion


class MAPE(Criterion):
    def __init__(self):
        super().__init__(name="MAPE", reduction='mean', classification_criterion=False)

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        return torch.mean(torch.abs((input_ - target) / target))

    def to_str(self, value: float) -> str:
        return "MAPE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
