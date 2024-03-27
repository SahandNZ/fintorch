import torch

from ._criterion import Criterion


class MSE(Criterion):
    def __init__(self):
        super().__init__(name="MSE", reduction='mean', classification_criterion=False)

    def _forward(self, input: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(input=input, target=target, reduction=self.reduction)

    def to_str(self, value: float) -> str:
        return "MSE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
