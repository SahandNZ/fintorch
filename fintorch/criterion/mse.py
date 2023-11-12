import torch

from fintorch.criterion.criterion import Criterion


class MSELoss(Criterion):
    def __init__(self):
        super().__init__(name="MSE", reduction='mean', classification_criterion=False)

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(input=input_, target=target, reduction=self.reduction)

    def to_str(self, value: float) -> str:
        return "MSE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
