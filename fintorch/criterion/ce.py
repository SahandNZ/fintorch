import torch
import torch.nn.functional as F

from fintorch.criterion.criterion import Criterion


class CELoss(Criterion):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(name="CE", reduction=reduction, classification_criterion=True)

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        weight = torch.sum(target, dim=0) / len(target)
        loss = F.cross_entropy(input=input_, target=target, weight=weight, reduction=self.reduction)

        return loss

    def to_str(self, value: float) -> str:
        return "CE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
