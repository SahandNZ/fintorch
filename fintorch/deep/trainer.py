import copy
import gc
import time
import warnings
from typing import Generator

import torch

from .cross_validation import CrossValidation
from .data_loader import DataLoader
from .dtype import Dataset, Epoch, Fold
from .lr_scheduler import LRScheduler
from .model import Model
from .optimizer import Optimizer
from .criterion import Criterion


class Trainer:
    def __init__(self, epochs_count: int, cross_validation: CrossValidation, data_loader: DataLoader,
                 criterion: Criterion, optimizer: Optimizer, lr_scheduler: LRScheduler,
                 gradient_clipping_threshold: float = None, auto_cuda: bool = True,
                 half_precision: bool = True) -> None:
        self.__epochs_count: int = epochs_count
        self.__cross_validation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LRScheduler = lr_scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold

        self.__auto_cuda: bool = auto_cuda
        self.__half_precision: bool = half_precision

    @property
    def epochs_count(self) -> int:
        return self.__epochs_count

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

    def optimize(self, dataset: Dataset, model: Model) -> Generator[Fold, None, None]:
        # move model to cuda if it's available
        model.to(self.device)

        for fold in self.cross_validation(dataset=dataset):
            model.reset()
            self.optimizer.reset(model=model)
            self.lr_scheduler.reset(optimizer=self.optimizer)

            fold.folds_count = self.cross_validation.folds_count
            fold.epochs_count = self.epochs_count

            for epoch_index in range(1, self.epochs_count + 1):
                epoch = Epoch(index=epoch_index, criterion=self.criterion)
                fold.epochs.append(epoch)

                for _ in self.__epoch_train_step(dataset, model, fold, epoch):
                    yield fold
                for _ in self.__epoch_val_step(dataset, model, fold, epoch):
                    yield fold
                for _ in self.__epoch_test_step(dataset, model, fold, epoch):
                    yield fold

            yield fold

        # move model back to cpu
        cpu = torch.device("cpu")
        model.to(cpu)

    def __epoch_train_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch) -> Generator:
        generator = self.___batched_common_step(dataset, model, fold, epoch, "train")
        for _ in generator:
            yield epoch

        epoch.model_state_dict = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}

    def __epoch_val_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch) -> Generator:
        generator = self.___batched_common_step(dataset, model, fold, epoch, "val")
        for _ in generator:
            yield

    def __epoch_test_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch) -> Generator:
        generator = self.___batched_common_step(dataset, model, fold, epoch, "test")
        for _ in generator:
            yield

    def ___batched_common_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch, mode: str) -> Generator:
        optimize = "train" == mode
        shuffle = not optimize
        timestamps = getattr(fold, f"{mode}_timestamps")
        iterator = self.data_loader(dataset=dataset, timestamps=timestamps, shuffle=shuffle)
        setattr(epoch, f"{mode}_batch_count", self.data_loader.batch_count)

        start_time = time.time()
        for index, (batch_x, batch_y) in enumerate(iterator):
            # move to cuda if it's available
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            loss = self.___common_step(model, batch_x, batch_y, optimize)

            # remove batch_x, batch_y
            del batch_x, batch_y

            getattr(epoch, f"{mode}_batch_losses").append(loss)
            getattr(epoch, f"{mode}_batch_times").append(time.time() - start_time)
            yield

            start_time = time.time()

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

    def ___common_step(self, model: Model, x: torch.Tensor, y: torch.Tensor, optimize: bool) -> torch.Tensor:
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

        return loss.detach().cpu().item()
