import torch
from torch import nn

from ._feed_forward import SFFeedForward
from ._model import SFModel


class SFAutoEncoder(SFModel):
    def __init__(
            self,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module,

            dim_input_sequence: int,
            dim_input_feature: int,
            dim_latent_feature: int,
            dim_latent_sequence: int,
            dim_output_sequence: int,
            dim_output_feature: int,

            num_hidden_layers: int,
    ) -> None:
        super().__init__(
            name="SF Auto Encoder",
            short_name="SF AE",
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
        )

        self.__encoder: SFFeedForward = SFFeedForward(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_latent_sequence,
            dim_output_feature=dim_latent_feature,

            num_hidden_layers=num_hidden_layers,
        )

        self.__decoder: SFFeedForward = SFFeedForward(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_latent_sequence,
            dim_input_feature=dim_latent_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
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
    def encoder(self) -> SFModel:
        return self.__encoder

    @property
    def decoder(self) -> SFModel:
        return self.__decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def _forward(self, x: torch.tensor) -> torch.Tensor:
        x_latent = self.encode(x)
        x_hat = self.decode(x_latent)
        return x_hat
