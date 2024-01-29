from typing import Tuple, Type

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


class DecoderModule(Module):
    def __init__(self, dataset: Dataset, encoder_module: EncoderModule, dim_output: int = 2, num_hidden_layers: int = 2,
                 batch_norm: bool = True, dropout: float = 0.5, model: Type[Model] = FeedForward,
                 auto_cuda: bool = True):
        model = model(
            dim_sequence=1,
            dim_feature=encoder_module.dim_latent,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        trainer = Trainer(
            cross_validation=SlidingWindowCrossValidation(train_percentage=80, dev_percentage=10, window_length=5000),
            data_loader=DataLoader(post_load_fn=self._post_load_fn),
            criterion=CE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-3),
            lr_scheduler=StepLR(step_size=1, gamma=0.9),
            gradient_clipping_threshold=1
        )

        super().__init__(dataset=dataset, model=model, trainer=trainer, auto_cuda=auto_cuda)
        self.__encoder_module: EncoderModule = encoder_module

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
        folds = self.trainer.optimize(dataset=self.dataset, model=self.model, epochs=epochs, batch_size=batch_size,
                                      progress=progress)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        latent_x = self.encoder_module.predict(x=x)
        return latent_x, y
