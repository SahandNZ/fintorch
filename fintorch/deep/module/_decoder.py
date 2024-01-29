import functools
import itertools
from typing import Dict, Tuple, Type

import torch
from rich.progress import Progress

from . import EncoderModule
from ._module import Module
from ..criterion import CE
from ..cross_validation import SlidingWindowCrossValidation
from ..data_loader import DataLoader
from ..dtype import Dataset
from ..lr_scheduler import StepLR
from ..model import FeedForward, Model
from ..optimizer import Adam
from ..trainer import Trainer
from ...enum import TimeFrame


class DecoderModule(Module):
    def __init__(self, dataset: Dataset, encoder_module: EncoderModule, num_hidden_layers: int = 2,
                 batch_norm: bool = True, dropout: float = 0.5, model_type: Type[Model] = FeedForward,
                 auto_cuda: bool = True):
        trainer = Trainer(
            cross_validation=SlidingWindowCrossValidation(train_percentage=80, dev_percentage=10, window_length=5000),
            data_loader=DataLoader(post_load_fn=self._post_load_fn),
            criterion=CE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-3),
            lr_scheduler=StepLR(step_size=1, gamma=0.9),
            gradient_clipping_threshold=1
        )

        super().__init__(dataset=dataset, trainer=trainer, auto_cuda=auto_cuda)
        self.__encoder_module: EncoderModule = encoder_module
        self.__model_type: Type[Model] = functools.partial(
            model_type,
            dim_sequence=1,
            dim_feature=encoder_module.dim_latent,
            dim_output=2,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout
        )

        self.__active_indices: Tuple[int, int] = None
        self.__models_dict: Dict[Tuple[str, TimeFrame], Model] = {}

    @property
    def encoder_module(self) -> EncoderModule:
        return self.__encoder_module

    @property
    def directory(self) -> str:
        return ""

    @property
    def path(self) -> str:
        return ""

    def optimize_and_store(self, epochs: int, batch_size: int, progress: Progress = None):
        symbol_values = list(range(len(self.dataset.symbols)))
        time_frame_values = list(range(len(self.dataset.time_frames)))
        for indices in itertools.product(symbol_values, time_frame_values):
            self.__active_indices = indices
            model = self._get_model()
            folds = self.trainer.optimize(dataset=self.dataset, model=model, epochs=epochs, batch_size=batch_size,
                                          progress=progress)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _get_model(self) -> Model:
        if self.__active_indices not in self.__models_dict:
            self.__models_dict[self.__active_indices] = self.__model_type()

        return self.__models_dict[self.__active_indices]

    def _post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        symbol_index, time_frame_index = self.__active_indices
        latent_x = self.encoder_module.predict(x=x)

        y = y[:, symbol_index, time_frame_index, 0]

        return latent_x, y
