import torch
import torch.nn.functional as F

from ._criterion import Criterion


class CE(Criterion):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(name="CE", reduction=reduction, classification_criterion=True)

    def _forward(self, input: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(input=input, target=target, reduction=self.reduction)

    def to_str(self, value: float) -> str:
        return "CE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
