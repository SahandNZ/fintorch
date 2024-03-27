import torch

from ._criterion import Criterion


class MNP(Criterion):
    def __init__(self, fee_rate: float = 2e-4):
        super().__init__(name="MNP", reduction='mean', classification_criterion=False)
        self.__fee_rate: float = fee_rate

    @property
    def fee_rate(self) -> float:
        return self.__fee_rate

    def _forward(self, input: torch.Tensor, target: torch.Tensor):
        pnl = torch.sum(target * input - self.fee_rate * torch.abs(input), dim=-1, keepdim=True)

        return pnl.mean()

    def to_str(self, value: float):
        return "MNP: {:.6f}".format(value)

    def compare(self, first: float, second: float):
        return first < second
