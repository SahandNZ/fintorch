import warnings
from typing import List

import torch
from torch import nn

from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
from .lr_scheduler import LRScheduler
from .metrics import Metrics
from .model import Model
from .optimizer import Optimizer
from .criterion import Criterion


class Trainer:
    def __init__(self, cross_validation: CrossValidation, data_loader: DataLoader, criterion: Criterion,
                 optimizer: Optimizer, lr_scheduler: LRScheduler, auto_cuda: bool = True):
        self.__cross_validation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LRScheduler = lr_scheduler

        self.__auto_cuda: bool = auto_cuda

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
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self.__lr_scheduler

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def device(self) -> torch.device:
        if self.auto_cuda and not torch.cuda.is_available():
            warnings.warn("Cuda device is not available while auto_cuda is True.")

        return torch.device('cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu')

    def optimize(self, dataset: Dataset, model: Model, epochs: int, batch_size: int) -> List[Fold]:
        folds = []
        for fold in self.cross_validation(dataset=dataset):
            self._prepare(model=model)
            for epoch in range(epochs):
                self._train(fold=fold, dataset=dataset, model=model, batch_size=batch_size)
                self._validation(fold=fold, dataset=dataset, model=model)
                self._test(fold=fold, dataset=dataset, model=model)

                # train_metrics = Metrics(epoch=epoch, criterion=self.criterion, outputs=train_outputs)
                # validation_metrics = Metrics(epoch=epoch, criterion=self.criterion, outputs=validation_outputs)
                # test_metrics = Metrics(epoch=epoch, criterion=self.criterion, outputs=test_outputs)
                #
                # fold.train_metrics_list.append(train_metrics)
                # fold.validation_metrics_list.append(validation_metrics)
                # fold.test_metrics_list.append(test_metrics)

            folds.append(fold)

        return folds

    def _prepare(self, model: Model):
        model.reset()
        self.optimizer.reset(model=model)
        self.lr_scheduler.reset(optimizer=self.optimizer)

        model.to(self.device)

    def _train(self, fold: Fold, dataset: Dataset, model: Model, batch_size: int):
        train_y, train_y_hat = [], []
        timestamps = fold.train_timestamps
        for batch_x, batch_y in self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=batch_size):
            batch_y, batch_y_hat = self._comment_step(model=model, x=batch_x, y=batch_y, optimize=True)
            train_y.append(batch_y)
            train_y_hat.append(batch_y_hat)

        train_y = torch.cat(train_y)
        train_y_hat = torch.cat(train_y_hat)

        return train_y, train_y_hat

    def _validation(self, fold: Fold, dataset: Dataset, model: Model):
        return self._validation_test_common_step(fold=fold, dataset=dataset, model=model, validation=True)

    def _test(self, fold: Fold, dataset: Dataset, model: Model):
        return self._validation_test_common_step(fold=fold, dataset=dataset, model=model, validation=False)

    def _validation_test_common_step(self, fold: Fold, dataset: Dataset, model: Model, validation: bool):
        timestamps = fold.validation_timestamps if validation else fold.test_timestamps
        x, y = next(iter(self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=len(timestamps))))
        y, y_hat = self._comment_step(model=model, x=x, y=y, optimize=False)
        return y, y_hat

    def _comment_step(self, model: Model, x: torch.Tensor, y: torch.Tensor, optimize: bool):
        # move to cuda
        x = x.to(self.device)
        y = y.to(self.device)

        if optimize:
            model.train()

            # forward prop
            with torch.autocast(device_type="cuda"):
                y_hat = model(x)
                loss = self.criterion(y_hat, y)

            # backward prop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(loss.item())
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(x)

        # move tensors to cpu
        x.cpu()
        y.cpu()

        return y.cpu(), y_hat.cpu()
