import torch
import torch.nn.functional as F

from fintorch.criterion.criterion import Criterion


class BCELoss(Criterion):
    def __init__(self, alpha: float = 0.5, reduction: str = 'mean'):
        super().__init__(name="BCE", reduction=reduction)
        self.alpha: float = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        if 0 <= self.alpha:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if 'mean' == self.reduction:
            loss = loss.mean()
        elif 'sum' == self.reduction:
            loss = loss.sum()

        return loss

    def to_str(self, value: float) -> str:
        return "BCE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
