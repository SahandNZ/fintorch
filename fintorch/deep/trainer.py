import copy
import gc
import time
import warnings
from typing import Callable, List

import torch
from rich.progress import Progress

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
                 optimizer: Optimizer, lr_scheduler: LRScheduler, gradient_clipping_threshold: float = None,
                 auto_cuda: bool = True, half_precision: bool = True,
                 log_fn: Callable[[Fold, int, int, float], None] = None):
        self.__cross_validation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LRScheduler = lr_scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold

        self.__auto_cuda: bool = auto_cuda
        self.__half_precision: bool = half_precision
        self.___log_fn: Callable[[Fold, int, int, float], None] = log_fn

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
    def gradient_clipping_threshold(self) -> float:
        return self.__gradient_clipping_threshold

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def device_type(self) -> str:
        if self.auto_cuda and not torch.cuda.is_available():
            warnings.warn("Cuda device is not available while auto_cuda is True.")

        return 'cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu'

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

    @property
    def half_precision(self) -> bool:
        return self.__half_precision

    @property
    def dtype(self) -> torch.dtype:
        if self.half_precision:
            return torch.float16 if 'cuda' == self.device_type else torch.bfloat16
        else:
            return torch.float32

    def optimize(self, dataset: Dataset, model: Model, epochs: int, batch_size: int,
                 progress: Progress = None) -> List[Fold]:
        folds = []
        for fold in self.cross_validation(dataset=dataset):
            self.__prepare(model=model)
            for epoch in range(1, epochs + 1):
                start_time = time.perf_counter()
                self.__train(dataset, model, batch_size, fold, epoch, progress)
                self.__validation(dataset, model, fold, epoch)
                self.__test(dataset, model, fold, epoch)
                epoch_time = time.perf_counter() - start_time
                self.__log_fn(fold, epoch, epochs, epoch_time)

            folds.append(fold)

        return folds

    def __prepare(self, model: Model):
        model.reset(device=self.device)
        self.optimizer.reset(model=model)
        self.lr_scheduler.reset(optimizer=self.optimizer)

    def __train(self, dataset: Dataset, model: Model, batch_size: int, fold: Fold, epoch: int,
                progress: Progress) -> None:
        if progress is not None:
            task = progress.add_task('Training on epoch {}'.format(epoch), total=len(dataset))

        y, y_hat = [], []
        timestamps = fold.train_timestamps
        for batch_x, batch_y in self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=batch_size):
            batch_y, batch_y_hat = self.__comment_step(model=model, x=batch_x, y=batch_y, optimize=True)
            y.append(batch_y)
            y_hat.append(batch_y_hat)

            if progress is not None:
                progress.update(task, advance=1)

        self.lr_scheduler.step()

        y = torch.cat(y)
        y_hat = torch.cat(y_hat)
        metrics = Metrics(criterion=self.criterion, epoch=epoch, y=y, y_hat=y_hat)
        fold.epoch_to_train_metrics[epoch] = metrics
        fold.epoch_to_model_state_dict[epoch] = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}

    def __validation(self, dataset: Dataset, model: Model, fold: Fold, epoch: int) -> None:
        self.__validation_test_common_step(fold=fold, dataset=dataset, model=model, epoch=epoch, validation=True)

    def __test(self, dataset: Dataset, model: Model, fold: Fold, epoch: int) -> None:
        self.__validation_test_common_step(fold=fold, dataset=dataset, model=model, epoch=epoch, validation=False)

    def __log_fn(self, fold: Fold, epoch: int, epochs: int, epoch_time: float) -> None:
        if self.___log_fn is not None:
            self.___log_fn(fold, epoch, epochs, epoch_time)

    def __validation_test_common_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, validation: bool):
        timestamps = fold.validation_timestamps if validation else fold.test_timestamps
        epoch_to_metrics = fold.epoch_to_validation_metrics if validation else fold.epoch_to_test_metrics

        x, y = next(iter(self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=len(timestamps))))
        y, y_hat = self.__comment_step(model=model, x=x, y=y, optimize=False)
        metrics = Metrics(criterion=self.criterion, epoch=epoch, y=y, y_hat=y_hat)
        epoch_to_metrics[epoch] = metrics

    def __comment_step(self, model: Model, x: torch.Tensor, y: torch.Tensor, optimize: bool):
        # move to cuda
        x = x.to(self.device)
        y = y.to(self.device)

        if optimize:
            model.train()

            # forward prop
            with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                y_hat = model(x)
                loss = self.criterion(y_hat, y)

            # backward prop
            loss.backward()

            if self.gradient_clipping_threshold is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping_threshold)

            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(x)

        # move to cpu
        cpu = torch.device("cpu")
        x = x.to(cpu)
        y = y.to(cpu)
        y_hat = y_hat.to(cpu)

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

        return y, y_hat
