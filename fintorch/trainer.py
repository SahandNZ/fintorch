import copy
from typing import Dict, List

import torch
from torch import nn
from tqdm.auto import tqdm

from fintorch.criterion.criterion import Criterion
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.cross_validation.fold import Fold
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.dataset import Dataset
from fintorch.metrics import Metrics
from fintorch.model.model import Model


class Trainer:
    def __init__(self,
                 epochs: int,
                 cross_validation: CrossValidation,
                 data_loader: DataLoader,
                 criterion: Criterion,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 gradient_clipping_threshold: float = None,
                 print_logs: bool = False,
                 show_progress_bar: bool = False,
                 show_learning_curve_plot: bool = False):

        self.__epochs: int = epochs
        self.__cross_validation: CrossValidation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: nn.Module = criterion
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold
        self.__print_logs: bool = print_logs
        self.__show_progress_bar: bool = show_progress_bar
        self.__show_learning_curve_plot: bool = show_learning_curve_plot

        self.__optimizer_initial_state_dict: Dict = copy.deepcopy(self.__optimizer.state_dict())
        self.__scheduler_initial_state_dict: Dict = copy.deepcopy(self.__scheduler.state_dict())

    @property
    def name(self) -> str:
        return self.criterion.name + '-' + self.optimizer.__class__.__name__

    @property
    def epochs(self) -> int:
        return self.__epochs

    @property
    def cross_validation(self) -> CrossValidation:
        return self.__cross_validation

    @property
    def data_loader(self) -> DataLoader:
        return self.__data_loader

    @property
    def criterion(self) -> Criterion:
        return self.__criterion

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self.__scheduler

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.__gradient_clipping_threshold

    @property
    def print_logs(self) -> bool:
        return self.__print_logs

    @property
    def show_progress_bar(self) -> bool:
        return self.__show_progress_bar

    @property
    def show_learning_curve_plot(self) -> bool:
        return self.__show_learning_curve_plot

    def reset(self, model: Model):
        model.reset()
        self.optimizer.load_state_dict(self.__optimizer_initial_state_dict)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.__optimizer_initial_state_dict)

    def common_step(self, x: torch.Tensor, y: torch.Tensor, model: Model, optimize: bool = False):
        if optimize:
            model.train()
            # forward prop
            y_hat = model(x)
            objective = self.criterion(y_hat, y)

            # backward prop
            objective.backward()
            if self.gradient_clipping_threshold:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.gradient_clipping_threshold)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(x)

        return y, y_hat

    def train(self, fold: Fold, model: Model):
        bar = range(self.epochs)
        if self.show_progress_bar:
            bar = tqdm(bar)
            bar.set_description("Train")

        for epoch in bar:
            self.data_loader.set_dataset(fold.train_set)

            epoch_metrics = Metrics(criterion=self.criterion, epoch=epoch)
            for batch_x, batch_y in self.data_loader:
                y, y_hat = self.common_step(x=batch_x, y=batch_y, model=model, optimize=True)
                epoch_metrics.append(y=y, y_hat=y_hat)
            epoch_metrics.set_model_state_dict(copy.deepcopy(model.state_dict()))
            fold.train_metrics_list.append(epoch_metrics)

            if self.scheduler is not None:
                self.__scheduler.step()

            if fold.best_train_metrics is None or fold.best_train_metrics < epoch_metrics:
                fold.best_train_metrics = epoch_metrics

            if self.show_progress_bar:
                learning_rate = next(iter(self.optimizer.param_groups))['lr']
                postfix = "current {} | best {} | LR: {:.6f}" \
                    .format(epoch_metrics, fold.best_train_metrics, learning_rate)
                bar.set_postfix_str(postfix)

    def validation(self, fold: Fold, model: Model):
        bar = fold.train_metrics_list
        if self.show_progress_bar:
            bar = tqdm(bar)
            bar.set_description("Validation")

        for train_metrics in bar:
            model.load_state_dict(train_metrics.model_state_dict)
            y, y_hat = self.common_step(x=fold.dev_set.x, y=fold.dev_set.y, model=model)
            epoch_metrics = Metrics(criterion=self.criterion, epoch=train_metrics.epoch, y=y, y_hat=y_hat)
            epoch_metrics.set_model_state_dict(train_metrics.model_state_dict)
            fold.dev_metrics_list.append(epoch_metrics)

            if fold.best_dev_metrics is None or fold.best_dev_metrics < epoch_metrics:
                fold.best_dev_metrics = epoch_metrics

            if self.show_progress_bar:
                postfix = "current {} | best {}".format(epoch_metrics, fold.best_dev_metrics)
                bar.set_postfix_str(postfix)

    def test(self, fold: Fold, model: Model):
        bar = fold.train_metrics_list
        if self.show_progress_bar:
            bar = tqdm(bar)
            bar.set_description("Test")

        for train_metrics in bar:
            model.load_state_dict(train_metrics.model_state_dict)
            y, y_hat = self.common_step(x=fold.test_set.x, y=fold.test_set.y, model=model)
            epoch_metrics = Metrics(criterion=self.criterion, epoch=train_metrics.epoch, y=y, y_hat=y_hat)
            epoch_metrics.set_model_state_dict(train_metrics.model_state_dict)
            fold.test_metrics_list.append(epoch_metrics)

            if fold.best_test_metrics is None or fold.best_test_metrics < epoch_metrics:
                fold.best_test_metrics = epoch_metrics

            if self.show_progress_bar:
                postfix = "current {} | best {}".format(epoch_metrics, fold.best_test_metrics)
                bar.set_postfix_str(postfix)

    def optimize(self, dataset: Dataset, model: Model) -> List[Fold]:
        folds = []
        self.reset(model)
        self.cross_validation.set_dataset(dataset)
        for fold in self.cross_validation:
            if self.__print_logs:
                print("#{} Fold".format(fold.index))

            self.train(fold=fold, model=model)
            self.validation(fold=fold, model=model)
            self.test(fold=fold, model=model)
            folds.append(fold)

            if self.show_learning_curve_plot:
                fold.show_learning_curve_plot()

        return folds
