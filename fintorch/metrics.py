from typing import Dict

import torch
from torch import nn

from fintorch.criterion.criterion import Criterion


class Metrics:
    def __init__(self, criterion: Criterion, epoch: int = None, y: torch.Tensor = None, y_hat: torch.Tensor = None,
                 probability: torch.Tensor = None):
        self.__epoch: int = epoch
        self.__model_state_dict: Dict = None
        self.__criterion: Criterion = criterion

        self.__y: torch.Tensor = y.detach().clone().cpu() if y is not None else torch.tensor([])
        self.__y_hat: torch.Tensor = y_hat.detach().clone().cpu() if y_hat is not None else torch.tensor([])
        self.__probability: torch.Tensor = probability.detach().clone().cpu() if probability is not None else None

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def model_state_dict(self) -> Dict:
        return self.__model_state_dict

    @property
    def criterion(self) -> Criterion:
        return self.__criterion

    @property
    def y(self) -> torch.Tensor:
        return self.__y

    @property
    def y_hat(self) -> torch.Tensor:
        return self.__y_hat

    @property
    def probability(self) -> torch.Tensor:
        if self.__probability is not None:
            return self.__probability
        else:
            probability, _ = torch.max(self.y_hat, dim=-1)
            return probability

    @property
    def objective(self):
        with torch.no_grad():
            return self.__criterion(self.y_hat, self.y) if 0 < len(self.y) and 0 < len(self.y_hat) else None

    @property
    def actual(self) -> torch.Tensor:
        if 1 < len(self.__y.shape) and 1 < self.__y.shape[1]:
            return torch.argmax(self.__y, dim=1)
        else:
            return self.__y

    @property
    def prediction(self) -> torch.Tensor:
        if 1 < len(self.__y_hat.shape) and 1 < self.__y_hat.shape[1]:
            return torch.argmax(self.__y_hat, dim=1)
        else:
            return self.__y_hat

    @property
    def mse_loss(self) -> float:
        return nn.functional.mse_loss(self.y_hat, self.y).detach().cpu().item()

    @property
    def mae_loss(self) -> float:
        return nn.functional.l1_loss(self.y_hat, self.y).detach().cpu().item()

    @property
    def accuracy(self) -> float:
        if self.actual is not None:
            rate = round((self.actual == self.prediction).sum().item() / len(self.actual) * 100, 2)
            return rate if 0 < len(self.actual) else 0

    @property
    def probability_accuracy(self) -> float:
        numerator = ((self.actual == self.prediction) * self.probability).sum().item()
        denominator = self.probability.sum().item()
        denominator = denominator if 0 < denominator else denominator + 1
        return round(numerator / denominator * 100, 2)

    def recall(self, label: int) -> float:
        numerator = ((label == self.actual) & (label == self.prediction)).sum().item()
        denominator = (label == self.actual).sum().item()
        denominator = denominator if 0 < denominator else denominator + 1
        return round(numerator / denominator * 100, 2)

    def precision(self, label: int) -> float:
        numerator = ((label == self.actual) & (label == self.prediction)).sum().item()
        denominator = (label == self.prediction).sum().item()
        denominator = denominator if 0 < denominator else denominator + 1
        return round(numerator / denominator * 100, 2)

    def probability_precision(self, label: int) -> float:
        numerator = (((label == self.actual) & (label == self.prediction)) * self.probability).sum().item()
        denominator = ((label == self.prediction) * self.probability).sum().item()
        denominator = denominator if 0 < denominator else denominator + 1
        return round(numerator / denominator * 100, 2)

    def f1(self, label: int) -> float:
        if self.actual is not None:
            r = self.recall(label)
            p = self.precision(label)
            numerator = 2 * r * p
            denominator = r + p
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator, 2)

    def append(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.detach().clone().cpu()
        y_hat = y_hat.detach().clone().cpu()

        self.__y = torch.cat([self.y, y], dim=0)
        self.__y_hat = torch.cat([self.y_hat, y_hat], dim=0)

    def set_model_state_dict(self, model_state_dict: Dict):
        self.__model_state_dict = model_state_dict

    def __str__(self):
        return self.__criterion.to_str(self.objective)

    def __add__(self, other):
        y = torch.cat([self.y, other.y], dim=0)
        y_hat = torch.cat([self.y_hat, other.y_hat], dim=0)
        metrics = Metrics(criterion=self.__criterion, y=y, y_hat=y_hat)

        return metrics

    def __lt__(self, other):
        return self.__criterion.less_than(self.objective, other.objective)
