from typing import Dict, List

import torch
from torch import nn

from fintorch.criterion.criterion import Criterion


class Metrics:
    def __init__(self, criterion: Criterion, epoch: int = None, y: torch.Tensor = None, y_hat: torch.Tensor = None):
        self.__criterion: Criterion = criterion
        self.__epoch: int = epoch

        # raw tensors
        self.__y: torch.Tensor = y.detach().clone().cpu() if y is not None else torch.tensor([])
        self.__y_hat: torch.Tensor = y_hat.detach().clone().cpu() if y_hat is not None else torch.tensor([])

        # model state dict
        self.__model_state_dict: Dict = None

        # classification properties
        self.__labels: List[int] = None

        # processed tensors
        self.__actual: torch.Tensor = None
        self.__prediction: torch.Tensor = None
        self.__probability: torch.Tensor = None

        # regression metrics
        self.__objective: float = None
        self.__mse_loss: float = None
        self.__mae_loss: float = None

        # classification metrics
        self.__accuracy: float = None
        self.__probability_accuracy: float = None
        self.__precisions_dict: Dict[int, float] = None
        self.__probability_precisions_dict: Dict[int, float] = None
        self.__recalls_dict: Dict[int, float] = None
        self.__probability_recalls_dict: Dict[int, float] = None
        self.__f1_scores_dict: Dict[int, float] = None
        self.__probability_f1_scores_dict: Dict[int, float] = None

    @staticmethod
    def get_best_metric(metrics_list: List):
        best_metric = None
        for metric in metrics_list:
            if best_metric is None or best_metric < metric:
                best_metric = metric

        return best_metric

    @staticmethod
    def aggregate(metrics_list: List):
        aggregated_metrics = Metrics(criterion=metrics_list[0].criterion)
        for metrics in metrics_list:
            aggregated_metrics.append(metrics.y, metrics.y_hat)

        return aggregated_metrics

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def model_state_dict(self) -> Dict:
        return self.__model_state_dict

    @model_state_dict.setter
    def model_state_dict(self, value: Dict):
        self.__model_state_dict = value

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
    def classification_task(self) -> bool:
        return self.criterion.classification_criterion

    @property
    def labels(self) -> List[int]:
        if self.classification_task and self.__labels is None:
            self.__labels = torch.unique(self.actual).tolist()

        return self.__labels

    @property
    def actual(self) -> torch.Tensor:
        if self.classification_task and self.__actual is None:
            self.__actual = self._get_label(self.__y)

        return self.__actual

    @property
    def prediction(self) -> torch.Tensor:
        if self.classification_task and self.__prediction is None:
            self.__prediction = self._get_label(self.__y_hat)

        return self.__prediction

    @property
    def probability(self) -> torch.Tensor:
        if self.classification_task and self.__probability is None:
            num_dims = len(self.y_hat.shape)
            last_dim = self.y_hat.shape[-1]
            if 2 == num_dims:
                if 1 < last_dim:
                    self.__probability = torch.argmax(self.y_hat, dim=1)
                else:
                    self.__probability = self.y_hat.view(-1)
            elif 1 == num_dims:
                self.__probability = self.y_hat.clone()

        return self.__probability

    @property
    def objective(self) -> float:
        if self.__objective is None:
            self.__objective = self.criterion(self.y_hat, self.y).detach().item()

        return self.__objective

    @property
    def mse_loss(self) -> float:
        if self.__mse_loss is None:
            self.__mse_loss = nn.functional.mse_loss(self.y_hat, self.y).detach().item()

        return self.__mse_loss

    @property
    def mae_loss(self) -> float:
        if self.__mae_loss is None:
            self.__mae_loss = nn.functional.l1_loss(self.y_hat, self.y).detach().item()

        return self.__mae_loss

    @property
    def accuracy(self) -> float:
        if self.classification_task and self.__accuracy is None and 0 < len(self.actual):
            self.__accuracy = round((self.actual == self.prediction).sum().item() / len(self.actual) * 100, 2)

        return self.__accuracy

    @property
    def probability_accuracy(self) -> float:
        if self.classification_task and self.__probability_accuracy is None and 0 < len(self.actual):
            numerator = ((self.actual == self.prediction) * self.probability).sum().item()
            denominator = self.probability.sum().item()
            denominator = denominator if 0 < denominator else denominator + 1
            self.__probability_accuracy = round(numerator / denominator * 100, 2)

        return self.__probability_accuracy

    def recall(self, label: int) -> float:
        if self.classification_task and self.__recalls_dict is None and 0 < len(self.actual):
            self.__recalls_dict = {}
            for _label in self.labels:
                numerator = ((_label == self.actual) & (_label == self.prediction)).sum().item()
                denominator = (_label == self.actual).sum().item()
                denominator = denominator if 0 < denominator else denominator + 1
                self.__recalls_dict[_label] = round(numerator / denominator * 100, 2)

        return self.__recalls_dict[label]

    def probability_recall(self, label: int) -> float:
        if self.classification_task and self.__probability_recalls_dict is None and 0 < len(self.actual):
            self.__probability_recalls_dict = {}
            for _label in self.labels:
                numerator = (((_label == self.actual) & (_label == self.prediction)) * self.probability).sum().item()
                denominator = ((_label == self.actual) * self.probability).sum().item()
                denominator = denominator if 0 < denominator else denominator + 1
                self.__probability_recalls_dict[_label] = round(numerator / denominator * 100, 2)

        return self.__probability_recalls_dict[label]

    def precision(self, label: int) -> float:
        if self.classification_task and self.__precisions_dict is None and 0 < len(self.actual):
            self.__precisions_dict = {}
            for _label in self.labels:
                numerator = ((_label == self.actual) & (_label == self.prediction)).sum().item()
                denominator = (_label == self.prediction).sum().item()
                denominator = denominator if 0 < denominator else denominator + 1
                self.__precisions_dict[_label] = round(numerator / denominator * 100, 2)

        return self.__precisions_dict[label]

    def probability_precision(self, label: int) -> float:
        if self.classification_task and self.__probability_precisions_dict is None and 0 < len(self.actual):
            self.__probability_precisions_dict = {}
            for _label in self.labels:
                numerator = (((_label == self.actual) & (_label == self.prediction)) * self.probability).sum().item()
                denominator = ((_label == self.prediction) * self.probability).sum().item()
                denominator = denominator if 0 < denominator else denominator + 1
                self.__probability_precisions_dict[_label] = round(numerator / denominator * 100, 2)

        return self.__probability_precisions_dict[label]

    def f1(self, label: int) -> float:
        if self.classification_task and self.__f1_scores_dict is None and 0 < len(self.actual):
            self.__f1_scores_dict = {}
            for _label in self.labels:
                r = self.recall(_label)
                p = self.precision(_label)
                numerator = 2 * r * p
                denominator = r + p
                denominator = denominator if 0 < denominator else denominator + 1
                self.__f1_scores_dict[_label] = round(numerator / denominator, 2)

        return self.__f1_scores_dict[label]

    def probability_f1(self, label: int) -> float:
        if self.classification_task and self.__probability_f1_scores_dict is None and 0 < len(self.actual):
            self.__probability_f1_scores_dict = {}
            for _label in self.labels:
                r = self.probability_recall(_label)
                p = self.probability_precision(_label)
                numerator = 2 * r * p
                denominator = r + p
                denominator = denominator if 0 < denominator else denominator + 1
                self.__probability_f1_scores_dict[_label] = round(numerator / denominator, 2)

        return self.__probability_f1_scores_dict[label]

    def append(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.detach().clone().cpu()
        y_hat = y_hat.detach().clone().cpu()

        self.__y = torch.cat([self.y, y], dim=0)
        self.__y_hat = torch.cat([self.y_hat, y_hat], dim=0)

        self._revoke_calculated_metrics()

    @staticmethod
    def _get_label(values: torch.Tensor):
        num_dims = len(values.shape)
        last_dim = values.shape[-1]
        if 2 == num_dims:
            if 1 < last_dim:
                return torch.argmax(values, dim=1)
            else:
                return values.view(-1)
        elif 1 == num_dims:
            return values

    def _revoke_calculated_metrics(self):
        self.__labels = None

        self.__actual = None
        self.__prediction = None
        self.__probability = None

        self.__objective = None
        self.__mse_loss = None
        self.__mae_loss = None

        self.__accuracy = None
        self.__probability_accuracy = None
        self.__recalls_dict = None
        self.__probability_recalls_dict = None
        self.__precisions_dict = None
        self.__probability_precisions_dict = None
        self.__f1_scores_dict = None
        self.__probability_f1_scores_dict = None

    def __str__(self):
        return self.criterion.to_str(self.objective)

    def __add__(self, other):
        y = torch.cat([self.y, other.y], dim=0)
        y_hat = torch.cat([self.y_hat, other.y_hat], dim=0)
        metrics = Metrics(criterion=self.__criterion, y=y, y_hat=y_hat)

        return metrics

    def __lt__(self, other):
        return self.__criterion.less_than(self.objective, other.objective)

    def __getstate__(self):
        dct = dict(self.__dict__)
        if "_Metrics__model_state_dict" in dct:
            del dct['_Metrics__model_state_dict']

        return dct
