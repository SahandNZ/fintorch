import copy
import gc
import time
import warnings
from typing import Generator, Tuple

import torch

from fintorch.deep.model import Model
from .criterion import Criterion
from .data_loader import DataLoader
from .dataset import Dataset
from .error import NanValueInBatchError
from .lr_scheduler import LrScheduler
from .optimizer import Optimizer
from .status import Epoch, Fold
from ..utils.hash import static_list_hash


class Trainer:
    def __init__(
            self,
            data_loader: DataLoader,
            criterion: Criterion,
            optimizer: Optimizer,
            lr_scheduler: LrScheduler,
            epochs_count: int,
            shuffle: bool,
            auto_cuda: bool,
            half_precision: bool,
            gradient_clipping_threshold: float,
    ) -> None:
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__lr_scheduler: LrScheduler = lr_scheduler

        self.__epochs_count: int = epochs_count
        self.__shuffle: bool = shuffle
        self.__auto_cuda: bool = auto_cuda
        self.__half_precision: bool = half_precision
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold

        self.__static_hash: int = static_list_hash([
            self.epochs_count,
            self.data_loader.static_hash,
            self.criterion.static_hash,
            self.optimizer.static_hash,
            self.lr_scheduler.static_hash,
            self.shuffle,
            self.gradient_clipping_threshold if self.gradient_clipping_threshold is not None else 1
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
    def shuffle(self) -> bool:
        return self.__shuffle

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.__gradient_clipping_threshold

    @property
    def half_precision(self) -> bool:
        return self.__half_precision

    @property
    def device_type(self) -> str:
        if self.auto_cuda and not torch.cuda.is_available():
            warnings.warn("Cuda device is not available while auto_cuda is True.")

        return 'cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu'

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_type)

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
        # move model to cuda device if it's available and prepare it for training
        model.to(self.device)
        model.reset()
        
        print(self.device)

        with self.optimizer, self.lr_scheduler:
            # reset model and optimizer and lr scheduler
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
                    
                # set it as a completed epoch
                epoch.completed = True

                # set usefull model state dicts
                if fold.best_train_epoch == epoch or fold.best_val_epoch == epoch or fold.best_test_epoch == epoch:
                    epoch.model_state_dict = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}

        # remove useless model state dict
        for epoch in fold.epochs:
            if fold.best_train_epoch != epoch and fold.best_val_epoch != epoch and fold.best_test_epoch != epoch:
                epoch.model_state_dict = {}

        # move model back to cpu
        cpu = torch.device("cpu")
        model.to(cpu)

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()


    def __epoch_train_step(self, dataset: Dataset, model: Model, fold: Fold, epoch: Epoch) -> Generator:
        generator = self.___batched_common_step(dataset, model, fold, epoch, "train")
        for _ in generator:
            yield

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
        shuffle = self.shuffle and optimize
        drop_nan = "test" == mode or "val" == mode
        timestamps = getattr(fold, f"{mode}_timestamps")
        iterator = self.data_loader(dataset=dataset, timestamps=timestamps, drop_nan=drop_nan, shuffle=shuffle)
        setattr(epoch, f"{mode}_batch_count", self.data_loader.batch_count)

        try:
            start_time = time.time()
            for index, (batch_x, batch_y, weight) in enumerate(iterator):
                # move to cuda if it's available
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                weight = weight.to(self.device)

                if 0 == len(batch_x) or 0 == len(batch_y):
                    continue

                if optimize and len(batch_x) < 2:
                    continue

                # calculate metrics
                batch_loss, batch_y_hat = self.___common_step(model, batch_x, batch_y, weight, optimize)
                batch_actual = torch.argmax(batch_y, dim=-1)
                batch_prediction = torch.argmax(batch_y_hat.detach(), dim=-1)
                batch_accuracy = (batch_actual == batch_prediction).sum() / len(batch_actual) * 100

                getattr(epoch, f"{mode}_batch_times").append(time.time() - start_time)
                getattr(epoch, f"{mode}_batch_losses").append(batch_loss.detach().cpu().item())
                getattr(epoch, f"{mode}_batch_accuracies").append(batch_accuracy.detach().cpu().item())
                yield

                # clean up memory
                del batch_x, batch_y, batch_loss, batch_y_hat, batch_actual, batch_prediction, batch_accuracy

                start_time = time.time()

        except NanValueInBatchError as e:
            if not drop_nan:
                raise e

    def ___common_step(
        self,
        model: Model,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        optimize: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if optimize:
            model.train()

            # forward prop
            if self.half_precision:
                with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                    y_hat = model(x)
                    loss = self.criterion(input=y_hat, target=y, weight=weight)
            else:
                y_hat = model(x)
                loss = self.criterion(input=y_hat, target=y, weight=weight)

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
                loss = self.criterion(input=y_hat, target=y, weight=weight)

        return loss, y_hat
