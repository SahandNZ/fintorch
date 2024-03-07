from typing import Type

import torch
from torch import nn

from fintorch.deep.model._feed_forward import FeedForward
from fintorch.deep.model._model import Model


class AutoEncoder(Model):
    def __init__(
            self,
            dim_sequence: int,
            dim_feature: int,
            dim_output: int,
            num_hidden_layers: int = 2,
            dropout: float = 0.5,
            batch_norm: bool = True,
            encoder: Type[Model] = FeedForward
    ) -> None:
        super().__init__(
            name="Auto Encoder",
            short_name="AE",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=nn.Tanh()
        )

        self.__encoder: Model = encoder(
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=self.activation_fn
        )

        self.__decoder: Model = FeedForward(
            dim_sequence=1,
            dim_feature=dim_output,
            dim_output=dim_sequence * dim_feature,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=self.activation_fn
        )

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
        encoded_x = encoded_x.unsqueeze(1)  # dims (Batch, Sequence (1), Latent feature)
        x_hat = self.decoder(encoded_x)
        x_hat = x_hat.view(x_hat.shape[0], self.dim_sequence, self.dim_feature)
        return x_hat

    def _forward(self, x: torch.tensor) -> torch.Tensor:
        encoded_x = self.encode(x)
        x_hat = self.decode(encoded_x)
        return x_hat
