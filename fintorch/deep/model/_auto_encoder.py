from typing import Type

import torch

from fintorch.deep.model._feed_forward import FeedForward
from fintorch.deep.model._model import Model


class AutoEncoder(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, dropout: float = 0,
                 encoder: Type[Model] = FeedForward) -> None:
        super().__init__(
            name="Auto Encoder",
            short_name="AE",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            dropout=dropout,
            activation_fn=None
        )
        self.__encoder: Model = encoder(dim_sequence=dim_sequence, dim_feature=dim_feature, dim_output=dim_output)
        self.__decoder: Model = FeedForward(dim_sequence=1, dim_feature=dim_output, dim_output=dim_sequence * dim_feature)

    @property
    def encoder(self) -> Model:
        return self.__encoder

    @property
    def decoder(self) -> Model:
        return self.__decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded_x = self.encoder(x)
        return encoded_x

    def decode(self, encoded_x: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(encoded_x)
        x_hat = x_hat.view(x_hat.shape[0], self.dim_sequence, self.dim_feature)
        return x_hat

    def forward(self, x: torch.tensor) -> torch.Tensor:
        encoded_x = self.encode(x)
        x_hat = self.decode(encoded_x)
        return x_hat
