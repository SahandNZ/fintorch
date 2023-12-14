import torch
import torch.nn.functional as F

from fintorch.criterion.criterion import Criterion


class CELoss(Criterion):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(name="CE", reduction=reduction, classification_criterion=True)

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        num_classes = target.shape[1]
        device_index = target.get_device()
        device = "cpu" if device_index < 0 else f"cuda:{device_index}"

        actual = torch.argmax(target, dim=-1)
        count = torch.tensor([(actual == label).sum() for label in range(num_classes)]).to(device)
        weight = count / len(actual)

        return F.cross_entropy(input=input_, target=target, weight=weight, reduction=self.reduction)

    def to_str(self, value: float) -> str:
        return "CE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
