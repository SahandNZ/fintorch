import torch

from fintorch.criterion.criterion import Criterion


class MSELoss(Criterion):
    def __init__(self):
        super().__init__(reduction='mean')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return torch.nn.functional.mse_loss(inputs, targets)

    def to_str(self, value: float) -> str:
        return "MSE: {:.6f}".format(value)

    def less_than(self, first: float, second: float) -> bool:
        return second < first
