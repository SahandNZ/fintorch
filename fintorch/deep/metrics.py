import copy
from typing import Dict, List

import numpy as np
import torch
from .criterion import Criterion
from torch import nn


class Metrics:
    def __init__(self, criterion: Criterion, epoch: int, y: torch.Tensor, y_hat: torch.Tensor):
        self.__criterion: Criterion = criterion
        self.__epoch: int = epoch

        self.__y: torch.Tensor = y.clone().detach().cpu()
        self.__y_hat: torch.Tensor = y_hat.clone().detach().cpu()
        self.__actual: torch.Tensor = torch.argmax(y, dim=-1) if self.classification_task else None
        self.__prediction: torch.Tensor = torch.argmax(y_hat, dim=-1) if self.classification_task else None
        self.__probability: torch.Tensor = torch.max(y_hat, dim=-1).values if self.classification_task else None

    @staticmethod
    def get_best_metric(metrics_list: List):
        best_metric = None
        for metric in metrics_list:
            if best_metric is None or best_metric < metric:
                best_metric = metric

        return best_metric

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
    def actual(self) -> torch.Tensor:
        return self.__actual

    @property
    def prediction(self) -> torch.Tensor:
        return self.__prediction

    @property
    def probability(self) -> torch.Tensor:
        return self.__probability

    @property
    def classification_task(self) -> bool:
        return self.criterion.classification_criterion

    @property
    def objective(self) -> float:
        return self.criterion(self.y_hat, self.y).detach().item()

    @property
    def mse_loss(self) -> float:
        return nn.functional.mse_loss(self.y_hat, self.y).detach().item()

    @property
    def mae_loss(self) -> float:
        return nn.functional.l1_loss(self.y_hat, self.y).detach().item()

    @property
    def accuracy(self) -> float:
        if self.classification_task and 0 < len(self.actual):
            return round((self.actual == self.prediction).sum().item() / len(self.actual) * 100, 2)

    @property
    def probability_accuracy(self) -> float:
        if self.classification_task and 0 < len(self.actual):
            numerator = ((self.actual == self.prediction) * self.probability).sum().item()
            denominator = self.probability.sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator * 100, 2)

    def recall(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            numerator = ((label == self.actual) & (label == self.prediction)).sum().item()
            denominator = (label == self.actual).sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator * 100, 2)

    def probability_recall(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            numerator = (((label == self.actual) & (label == self.prediction)) * self.probability).sum().item()
            denominator = ((label == self.actual) * self.probability).sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator * 100, 2)

    def precision(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            numerator = ((label == self.actual) & (label == self.prediction)).sum().item()
            denominator = (label == self.prediction).sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator * 100, 2)

    def probability_precision(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            numerator = (((label == self.actual) & (label == self.prediction)) * self.probability).sum().item()
            denominator = ((label == self.prediction) * self.probability).sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator * 100, 2)

    def f1(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            r = self.recall(label)
            p = self.precision(label)
            numerator = 2 * r * p
            denominator = r + p
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator, 2)

    def probability_f1(self, label: int) -> float:
        if self.classification_task and 0 < len(self.actual):
            r = self.probability_recall(label)
            p = self.probability_precision(label)
            numerator = 2 * r * p
            denominator = r + p
            denominator = denominator if 0 < denominator else denominator + 1
            return round(numerator / denominator, 2)

    def __str__(self):
        return self.criterion.to_str(self.objective)

    def __lt__(self, other):
        return self.__criterion.less_than(self.objective, other.objective)
