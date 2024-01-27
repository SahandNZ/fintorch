import copy
import gc
import time
import warnings
from datetime import datetime
from typing import List

import torch
from rich.progress import Progress

from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Fold
from .lr_scheduler import LRScheduler
from .model import Model
from .optimizer import Optimizer
from .criterion import Criterion
from ..utils.memory import get_memory_status


class Trainer:
    def __init__(self, cross_validation: CrossValidation, data_loader: DataLoader, criterion: Criterion,
                 optimizer: Optimizer, lr_scheduler: LRScheduler, gradient_clipping_threshold: float = None,
                 auto_cuda: bool = True, half_precision: bool = True) -> None:
        self.__cross_validation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LRScheduler = lr_scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold

        self.__auto_cuda: bool = auto_cuda
        self.__half_precision: bool = half_precision

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
            self.__prepare(model=model, fold=fold)
            for epoch in range(1, epochs + 1):
                start_time = time.perf_counter()
                self.__train_step(dataset, model, fold, epoch, batch_size, progress)
                self.__val_step(dataset, model, fold, epoch, batch_size, progress)
                self.__test_step(dataset, model, fold, epoch, batch_size, progress)
                epoch_time = time.perf_counter() - start_time
                self.__logger_fn(fold, epoch, epochs, epoch_time)

            folds.append(fold)

        return folds

    def __prepare(self, model: Model, fold: Fold) -> None:
        model.reset()
        fold.criterion = self.criterion
        self.optimizer.reset(model=model)
        self.lr_scheduler.reset(optimizer=self.optimizer)

    def __train_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, batch_size: int,
                     progress: Progress) -> None:
        timestamps = fold.train_timestamps
        loss = self.___batched_common_step(dataset=dataset, model=model, timestamps=timestamps, epoch=epoch,
                                           batch_size=batch_size, optimize=True, validation=False, progress=progress)
        fold.epoch_to_train_loss[epoch] = loss
        fold.epoch_to_model_state_dict[epoch] = copy.deepcopy(model.state_dict())

    def __val_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, batch_size: int,
                   progress: Progress) -> None:
        self.___val_and_test_common_step(dataset=dataset, model=model, fold=fold, epoch=epoch, batch_size=batch_size,
                                         progress=progress, validation=True)

    def __test_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, batch_size: int,
                    progress: Progress) -> None:
        self.___val_and_test_common_step(dataset=dataset, model=model, fold=fold, epoch=epoch, batch_size=batch_size,
                                         progress=progress, validation=False)

    def __logger_fn(self, fold: Fold, epoch: int, epochs: int, epoch_time: float) -> None:
        elapsed_time = epoch_time * epoch
        total_time = elapsed_time * (epochs / epoch)
        remaining_time = total_time - elapsed_time
        elapsed_time_str = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
        remaining_time_str = datetime.strftime(datetime.utcfromtimestamp(remaining_time), '%H:%M:%S')
        total_time_str = datetime.strftime(datetime.utcfromtimestamp(total_time), '%H:%M:%S')

        # logs
        print("Epoch ({}/{}) (elapsed: {} remaining: {} total: {}) (lr: {:.8f})"
              .format(epoch, epochs, elapsed_time_str, remaining_time_str, total_time_str, self.optimizer.lr))
        print("\t- Metrics")
        print("\t\t- {:<12} {}: {:.4f} (best: {:.4f})"
              .format("Train", self.criterion.name, fold.epoch_to_train_loss[epoch], fold.best_train_loss))
        print("\t\t- {:<12} {}: {:.4f} (best: {:.4f})"
              .format("Validation", self.criterion.name, fold.epoch_to_validation_loss[epoch],
                      fold.best_validation_loss))
        print("\t\t- {:<12} {}: {:.4f} (best: {:.4f}) (best validation: {:.4f})"
              .format("Test", self.criterion.name, fold.epoch_to_test_loss[epoch], fold.best_test_loss,
                      fold.best_validation_on_test_loss))
        print(get_memory_status(start="\t- "))

    def ___val_and_test_common_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: int, batch_size: int,
                                    progress: Progress, validation: bool) -> None:
        timestamps = fold.validation_timestamps if validation else fold.test_timestamps
        loss = self.___batched_common_step(dataset=dataset, model=model, timestamps=timestamps, epoch=epoch,
                                           batch_size=batch_size, optimize=False, validation=validation,
                                           progress=progress)

        epoch_to_loss = fold.epoch_to_validation_loss if validation else fold.epoch_to_test_loss
        epoch_to_loss[epoch] = loss

    def ___batched_common_step(self, dataset: Dataset, model: Model, timestamps: List[int], epoch: int, batch_size: int,
                               optimize: bool, validation: bool, progress: Progress) -> float:
        iterator = self.data_loader(dataset=dataset, timestamps=timestamps, batch_size=batch_size)

        if progress is not None:
            if optimize:
                description = "Training step of epoch {}".format(epoch)
            elif validation:
                description = "Validation step of epoch {}".format(epoch)
            else:
                description = "Test step of epoch {}".format(epoch)
            task = progress.add_task(description=description, total=self.data_loader.batch_count)

        # move model to cuda if it's available
        model.to(self.device)

        total_loss = 0
        total_items = 0
        for index, (batch_x, batch_y) in enumerate(iterator):
            # move to cuda if it's available
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_loss = self.___common_step(model=model, x=batch_x, y=batch_y, optimize=optimize)
            batch_loss = batch_loss.detach().cpu().item()

            total_loss = (total_loss * total_items + batch_loss * batch_size) / (total_items + batch_size)
            total_items += batch_size

            # update rich progress bar
            if progress is not None:
                progress.update(task, advance=1)

            # remove batch_x, batch_y, batch_loss
            del batch_x, batch_y, batch_loss

        if optimize and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if progress is not None:
            progress.update(task, visible=False)

        # move model to cpu
        model.to(torch.device("cpu"))

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

        return total_loss

    def ___common_step(self, model: Model, x: torch.Tensor, y: torch.Tensor, optimize: bool):
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
                loss = self.criterion(y_hat, y)

        return loss
