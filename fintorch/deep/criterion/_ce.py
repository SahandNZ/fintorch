import torch
import torch.nn.functional as F

from ._criterion import Criterion


class CE(Criterion):
    def __init__(self, label_smoothing: float, reduction: str = 'mean'):
        super().__init__(name="CE", label_smoothing=label_smoothing, reduction=reduction, classification_criterion=True)

    def _forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(
            input=input,
            target=target,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )
        
        return loss

    def to_str(self, value: float) -> str:
        return "CE: {:.6f}".format(value)

    def compare(self, first: float, second: float) -> bool:
        return second < first
