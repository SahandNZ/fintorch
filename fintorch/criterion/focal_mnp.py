import torch

from fintorch.criterion.criterion import Criterion


class FocalMNPLoss(Criterion):
    def __init__(self, gamma: float = 0, fee_rate: float = 2e-4):
        super().__init__(name="Focal MNP", reduction='mean', classification_criterion=False)
        self.__gamma: float = gamma
        self.__fee_rate: float = fee_rate

    @property
    def gamma(self) -> float:
        return self.__gamma

    @property
    def fee_rate(self) -> float:
        return self.__fee_rate

    def forward(self, qty: torch.Tensor, roc: torch.Tensor):
        if 1 < len(qty.shape):
            qty = qty.view(-1)

        pnl = roc * qty - self.fee_rate * torch.abs(qty)
        grade_of_confidence = (qty * torch.sign(roc) + 1) / 2
        focal_pnl = (1 - grade_of_confidence) ** self.gamma * pnl

        return focal_pnl.mean()

    def to_str(self, value: float):
        return "Focal MNP: {:.6f}".format(value)

    def less_than(self, first: float, second: float):
        return first < second
