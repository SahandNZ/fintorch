import torch
import torch.nn.functional as F

from ._criterion import Criterion


class FocalCE(Criterion):
    def __init__(self, gamma: float = 2, reduction: str = 'mean'):
        super().__init__(name="Focal CE", reduction=reduction, classification_criterion=True)
        self.gamma: float = gamma

    def _forward(self, input: torch.Tensor, target: torch.Tensor):
        loss = F.cross_entropy(input=input, target=target, reduction="none")

        actual = torch.argmax(target, dim=-1)
        p_t = input[range(len(input)), actual]
        loss = loss * ((1 - p_t) ** self.gamma)

        if 'mean' == self.reduction:
            loss = loss.mean()
        elif 'sum' == self.reduction:
            loss = loss.sum()

        return loss

    def to_str(self, value: float) -> str:
        return "Focal CE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
