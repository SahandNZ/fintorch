from typing import List, Tuple, Type

import torch

from ._module import Module
from ..criterion import MSE
from ..criterion._mape import MAPE
from ..cross_validation import CrossValidation
from ..data_loader import DataLoader
from ..dtype import Dataset
from ..lr_scheduler import StepLR
from ..model import Model
from ..model import AutoEncoder
from ..optimizer import Adam
from ..trainer import Trainer


class EncoderModule(Module):
    def __init__(self, dim_symbol: int, dim_time_frame: int, dim_sequence: int, dim_feature, dim_latent: int,
                 model: Type[Model]):
        trainer = Trainer(
            cross_validation=CrossValidation(train_percentage=80, dev_percentage=10),
            data_loader=DataLoader(post_load_fn=self._post_load_fn),
            criterion=MSE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-4),
            lr_scheduler=StepLR(step_size=5, gamma=0.9)
        )
        super().__init__(trainer=trainer)

        self.__dim_symbol: int = dim_symbol
        self.__dim_time_frame: int = dim_time_frame
        self.__dim_sequence: int = dim_sequence
        self.__dim_feature: int = dim_feature
        self.__dim_latent: int = dim_latent

        dim_flat_feature = dim_symbol * dim_time_frame * dim_feature
        self.__auto_encoder: AutoEncoder = AutoEncoder(dim_sequence=dim_sequence, dim_feature=dim_flat_feature,
                                                       dim_output=dim_latent, encoder=model)

    @property
    def dim_symbol(self) -> int:
        return self.__dim_symbol

    @property
    def dim_time_frame(self) -> int:
        return self.__dim_time_frame

    @property
    def dim_sequence(self) -> int:
        return self.__dim_sequence

    @property
    def dim_feature(self) -> int:
        return self.__dim_feature

    @property
    def dim_latent(self) -> int:
        return self.__dim_latent

    @property
    def auto_encoder(self) -> AutoEncoder:
        return self.__auto_encoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bl = self.auto_encoder.encode(x)  # dims (Batch, Latent Feature)
        return bl

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        bf = self.auto_encoder.decode(x)  # dims (Batch, Sequence * Feature)
        bsf_hat = bf.view(bf.shape[0], self.dim_sequence, -1)
        return bsf_hat

    def forward(self, x: torch.tensor) -> torch.tensor:
        latent_x = self.encode(x)
        x_hat = self.decode(latent_x)
        return x_hat

    def optimize(self, dataset: Dataset, epochs: int, batch_size: int):
        self.trainer.optimize(dataset=dataset, model=self.auto_encoder, epochs=epochs, batch_size=batch_size)

    def predict(self, dataset: Dataset, timestamps: List[int]) -> List:
        pass

    def _post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        bsatf = x.permute(0, 3, 1, 2, 4).contiguous()  # dims (Batch, Sequence, Asset, Time frame, Feature)
        bsf = bsatf.view(bsatf.shape[0], bsatf.shape[1], -1)  # dims (Batch, Sequence, Asset * Time frame * Feature)
        return bsf, bsf
