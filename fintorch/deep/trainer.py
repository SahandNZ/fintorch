import copy
import gc
import time
import warnings
from typing import Generator, Tuple

import torch

from .criterion import Criterion
from .data_loader import DataLoader
from .dtype import Dataset, Epoch, Fold
from .lr_scheduler import LrScheduler
from .model import Model
from .optimizer import Optimizer
from ..utils.hash import static_list_hash


class Trainer:
    def __init__(
            self,
            epochs_count: int,
            data_loader: DataLoader,
            criterion: Criterion,
            optimizer: Optimizer,
            lr_scheduler: LrScheduler,
            gradient_clipping_threshold: float = None,
            auto_cuda: bool = True,
            half_precision: bool = True
    ) -> None:

        self.__epochs_count: int = epochs_count
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LrScheduler = lr_scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold

        self.__auto_cuda: bool = auto_cuda
        self.__half_precision: bool = half_precision

        self.__static_hash: int = static_list_hash([
            self.epochs_count,
            self.data_loader.static_hash,
            self.criterion.static_hash,
            self.optimizer.static_hash,
            self.lr_scheduler.static_hash,
            self.gradient_clipping_threshold
        ])

    @property
    def epochs_count(self) -> int:
        return self.__epochs_count

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
    def lr_scheduler(self) -> LrScheduler:
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

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def optimize_fold(self, dataset: Dataset, model: Model, fold: Fold) -> Generator[None, None, None]:
        # move model to cuda device if it's available
        model.to(self.device)

        for _ in self.__fold_step(dataset=dataset, model=model, fold=fold):
            yield

        # move model back to cpu
        cpu = torch.device("cpu")
        model.to(cpu)

    def __fold_step(self, dataset: Dataset, model: Model, fold: Fold) -> Generator[None, None, None]:
        # reset model and optimizer and lr scheduler
        model.reset()
        self.optimizer.reset(model=model)
        self.lr_scheduler.reset(optimizer=self.optimizer)

        # epoch loop
        fold.epochs_count = self.epochs_count
        for epoch_index in range(1, self.epochs_count + 1):
            epoch = Epoch(index=epoch_index, criterion=self.criterion)
            fold.epochs.append(epoch)

            for _ in self.__epoch_train_step(dataset, model, fold, epoch):
                yield
            for _ in self.__epoch_val_step(dataset, model, fold, epoch):
                yield
            for _ in self.__epoch_test_step(dataset, model, fold, epoch):
                yield

            epoch.completed = True

    def __epoch_train_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch) -> Generator:
        generator = self.___batched_common_step(dataset, model, fold, epoch, "train")
        for _ in generator:
            yield

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

            if 0 == len(batch_x) or 0 == len(batch_y):
                continue

            batch_loss, batch_y_hat = self.___common_step(model, batch_x, batch_y, optimize)
            batch_actual = torch.argmax(batch_y, dim=-1)
            batch_prediction = torch.argmax(batch_y_hat, dim=-1)
            batch_accuracy = (batch_actual == batch_prediction).sum() / len(batch_y) * 100

            # remove batch_x, batch_y, batch_y_hat
            del batch_x, batch_y, batch_y_hat

            getattr(epoch, f"{mode}_batch_times").append(time.time() - start_time)
            getattr(epoch, f"{mode}_batch_losses").append(batch_loss.clone().cpu().item())
            getattr(epoch, f"{mode}_batch_accuracies").append(batch_accuracy.cpu().item())
            yield

            start_time = time.time()

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

    def ___common_step(self, model: Model, x: torch.Tensor, y: torch.Tensor, optimize: bool) \
            -> Tuple[torch.Tensor, torch.Tensor]:
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

        return loss, y_hat
