import torch
import torch.nn.functional as F

from fintorch.criterion.criterion import Criterion


class CELoss(Criterion):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(name="CE", reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        actual = torch.argmax(target, dim=-1)
        _, counts = actual.unique(return_counts=True)
        weight = counts / len(actual)
        return F.cross_entropy(input=input, target=target, weight=weight, reduction=self.reduction)

    def to_str(self, value: float) -> str:
        return "CE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
