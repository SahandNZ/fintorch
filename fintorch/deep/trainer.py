import copy
import gc
import time
import warnings
from typing import Callable, List, Tuple

import torch
from rich.progress import Progress, Task, TaskID

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
        timestamps = fold.train_timestamps
        y, y_hat = self.__batched_common_step(dataset=dataset, model=model, timestamps=timestamps, epoch=epoch,
                                              batch_size=batch_size, optimize=True, progress=progress)

        metrics = Metrics(criterion=self.criterion, epoch=epoch, y=y, y_hat=y_hat)
        fold.epoch_to_train_metrics[epoch] = metrics
        fold.epoch_to_model_state_dict[epoch] = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}

    def __validation(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, progress: Progress) -> None:
        self.__val_and_test_common_step(dataset=dataset, model=model, fold=fold, epoch=epoch, validation=True,
                                        progress=progress)

    def __test(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, progress: Progress) -> None:
        self.__val_and_test_common_step(dataset=dataset, model=model, fold=fold, epoch=epoch, validation=False,
                                        progress=progress)

    def __log_fn(self, fold: Fold, epoch: int, epochs: int, epoch_time: float) -> None:
        if self.___log_fn is not None:
            self.___log_fn(fold, epoch, epochs, epoch_time)

    def __val_and_test_common_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, validation: bool,
                                   progress: Progress):
        timestamps = fold.validation_timestamps if validation else fold.test_timestamps
        y, y_hat = self.__batched_common_step(dataset=dataset, model=model, timestamps=timestamps, epoch=epoch,
                                              batch_size=1024, optimize=False, progress=progress)

        epoch_to_metrics = fold.epoch_to_validation_metrics if validation else fold.epoch_to_test_metrics
        metrics = Metrics(criterion=self.criterion, epoch=epoch, y=y, y_hat=y_hat)
        epoch_to_metrics[epoch] = metrics

    def __batched_common_step(self, dataset: Dataset, model: Model, timestamps: List[int], epoch: int, batch_size: int,
                              optimize: bool, progress: Progress = None) -> Tuple[torch.Tensor, torch.Tensor]:
        y, y_hat = [], []
        batch_iterator = self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=batch_size)

        if progress is not None:
            description = "Training epoch {}".format(epoch) if optimize else "Evaluating epoch {}".format(epoch)
            task = progress.add_task(description=description, total=self.data_loader.batch_count)

        for batch_x, batch_y in batch_iterator:
            batch_y, batch_y_hat = self.__comment_step(model=model, x=batch_x, y=batch_y, optimize=optimize)
            y.append(batch_y)
            y_hat.append(batch_y_hat)

            if progress is not None:
                progress.update(task, advance=1)

        if optimize:
            self.lr_scheduler.step()

        y = torch.cat(y)
        y_hat = torch.cat(y_hat)

        return y, y_hat

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
