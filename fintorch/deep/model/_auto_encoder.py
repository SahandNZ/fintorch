import torch
from torch import nn

from fintorch.deep.model._feed_forward import FeedForward
from fintorch.deep.model._model import Model


class AutoEncoder(Model):
    def __init__(
            self,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_latent_sequence: int,
            dim_latent_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,
            num_hidden_layers: int,
            dropout: float,
            batch_norm: bool
    ) -> None:
        super().__init__(
            name="Auto Encoder",
            short_name="AE",
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=nn.Tanh()
        )

        self.__encoder: FeedForward = FeedForward(
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_latent_sequence,
            dim_output_feature=dim_latent_feature,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=self.activation_fn
        )

        self.__decoder: FeedForward = FeedForward(
            dim_input_sequence=dim_latent_sequence,
            dim_input_feature=dim_latent_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=self.activation_fn
        )

        self.__dim_latent_sequence: int = dim_latent_sequence
        self.__dim_latent_feature: int = dim_latent_feature

    @property
    def dim_latent_sequence(self) -> int:
        return self.__dim_latent_sequence

    @property
    def dim_latent_feature(self) -> int:
        return self.__dim_latent_feature

    @property
    def encoder(self) -> Model:
        return self.__encoder

    @property
    def decoder(self) -> Model:
        return self.__decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def _forward(self, x: torch.tensor) -> torch.Tensor:
        x_latent = self.encode(x)
        x_hat = self.decode(x_latent)
        return x_hat
