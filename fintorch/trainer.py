import copy
import gc
import warnings
from typing import List

import torch
from tqdm.auto import tqdm

from fintorch.criterion.criterion import Criterion
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.cross_validation.fold import Fold
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.dataset import Dataset
from fintorch.lr_scheduler import LRScheduler
from fintorch.metrics import Metrics
from fintorch.model.model import Model
from fintorch.optimizer import Optimizer
from fintorch.utils.memory import get_memory_status


class Trainer:
    def __init__(
            self,
            epochs: int,
            cross_validation: CrossValidation,
            data_loader: DataLoader,
            criterion: Criterion,
            optimizer: Optimizer,
            scheduler: LRScheduler = None,
            gradient_clipping_threshold: float = None,
            auto_cuda: bool = True,
            print_logs: bool = False,
            show_progress_bar: bool = False,
            print_memory_status_logs: bool = False,
            show_learning_curve_plot: bool = False,
            print_classification_logs: bool = False,
    ):

        self.__epochs: int = epochs
        self.__cross_validation: CrossValidation = cross_validation
        self.__data_loader: DataLoader = data_loader
        self.__criterion: Criterion = criterion
        self.__optimizer: Optimizer = optimizer
        self.__scheduler: LRScheduler = scheduler
        self.__gradient_clipping_threshold: float = gradient_clipping_threshold
        self.__auto_cuda: bool = auto_cuda
        self.__print_logs: bool = print_logs
        self.__show_progress_bar: bool = show_progress_bar
        self.__print_memory_status_logs: bool = print_memory_status_logs
        self.__show_learning_curve_plot: bool = show_learning_curve_plot
        self.__print_classification_logs: bool = print_classification_logs

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
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def scheduler(self) -> LRScheduler:
        return self.__scheduler

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.__gradient_clipping_threshold

    @property
    def auto_cuda(self) -> bool:
        return self.__auto_cuda

    @property
    def device(self) -> str:
        if self.auto_cuda and not torch.cuda.is_available():
            warnings.warn("Cuda device is not available while auto_cuda is active.")
        device = torch.device('cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu')

        return device

    @property
    def print_logs(self) -> bool:
        return self.__print_logs

    @property
    def show_progress_bar(self) -> bool:
        return self.__show_progress_bar

    @property
    def print_memory_status_logs(self) -> bool:
        return self.__print_memory_status_logs

    @property
    def show_learning_curve_plot(self) -> bool:
        return self.__show_learning_curve_plot

    @property
    def print_classification_logs(self) -> bool:
        return self.__print_classification_logs

    def _common_step(self, x: torch.Tensor, y: torch.Tensor, model: Model, optimize: bool = False):
        # move to tensors and model cuda if it's available
        model.to(self.device)
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
            if self.gradient_clipping_threshold:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.gradient_clipping_threshold)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            model.eval()
            with torch.no_grad():
                y_hat = model(x)

        # move tensors to cpu
        x.cpu()
        y.cpu()

        return y.cpu(), y_hat.cpu()

    def _val_test_common_step(self, fold: Fold, model: Model, validation: bool):
        dataset = fold.dev_set if validation else fold.test_set

        bar = fold.train_metrics_list
        if self.show_progress_bar:
            bar_description = "Validation" if validation else "Test"
            bar = tqdm(bar, desc=bar_description)

        for train_metrics in bar:
            model.load_state_dict(train_metrics.model_state_dict)
            y, y_hat = self._common_step(x=dataset.x, y=dataset.y, model=model)
            epoch_metrics = Metrics(criterion=train_metrics.criterion, epoch=train_metrics.epoch, y=y, y_hat=y_hat)
            epoch_metrics.model_state_dict = train_metrics.model_state_dict
            fold.append_dev_metrics(epoch_metrics) if validation else fold.append_test_metrics(epoch_metrics)

            if self.show_progress_bar:
                best_metrics = fold.best_dev_metrics if validation else fold.best_test_metrics
                bar.set_postfix_str("current {} | best {}".format(epoch_metrics, best_metrics))

    def _reset(self, model: Model):
        model.reset()
        self.optimizer.reset(model=model)
        if self.scheduler is not None:
            self.scheduler.reset(self.optimizer)

    def pre_logs(self, fold: Fold, model: Model):
        if self.print_logs:
            print("#{} Fold".format(fold.index))

            if self.print_memory_status_logs:
                print(get_memory_status(start="\t"))

    def prepare(self, fold: Fold, model: Model):
        self._reset(model=model)

    def train(self, fold: Fold, model: Model):
        bar = range(self.epochs)
        if self.show_progress_bar:
            bar = tqdm(bar)
            bar.set_description("Train")

        for epoch in bar:
            epoch_metrics = Metrics(criterion=self.criterion, epoch=epoch)
            self.data_loader.set_dataset(fold.train_set)
            for batch_x, batch_y in self.data_loader:
                y, y_hat = self._common_step(x=batch_x, y=batch_y, model=model, optimize=True)
                epoch_metrics.append(y=y, y_hat=y_hat)

            cpu_state_dict = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
            epoch_metrics.model_state_dict = cpu_state_dict
            fold.append_train_metrics(epoch_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.show_progress_bar:
                best_metric = fold.best_train_metrics
                postfix = "current {} | best {} | LR: {:.6f}".format(epoch_metrics, best_metric, self.optimizer.lr)
                bar.set_postfix_str(postfix)

    def validation(self, fold: Fold, model: Model):
        self._val_test_common_step(fold=fold, model=model, validation=True)

    def test(self, fold: Fold, model: Model):
        self._val_test_common_step(fold=fold, model=model, validation=False)

    def free_memory(self, fold: Fold, model: Model):
        device = "cpu"
        fold.to(device)
        model.to(device)

        gc.collect()
        torch.cuda.empty_cache()

    def post_logs(self, fold: Fold, model: Model):
        if self.print_logs:
            if self.print_memory_status_logs:
                print(get_memory_status(start="\t"))

            if self.show_learning_curve_plot:
                fold.show_learning_curve_plot()

            if self.print_classification_logs:
                fold.print_classification_logs()

    def optimize(self, dataset: Dataset, model: Model) -> List[Fold]:
        folds = []
        self.cross_validation.set_dataset(dataset)

        bar = list(self.cross_validation)
        if self.show_progress_bar:
            bar = tqdm(bar, desc="Cross Validation")

        for fold in bar:
            folds.append(fold)

            self.pre_logs(fold=fold, model=model)
            self.prepare(fold=fold, model=model)
            self.train(fold=fold, model=model)
            self.validation(fold=fold, model=model)
            self.test(fold=fold, model=model)
            self.free_memory(fold=fold, model=model)
            self.post_logs(fold=fold, model=model)

        overall_fold = Fold.aggregate(folds)
        return folds, overall_fold
