from torch import nn

from ._model import TSFModel
from ..sf import SFGRU


class GRU(TSFModel):
    def __init__(
            self,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module,

            dim_input_time_frame: int,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_latent_sequence: int,
            dim_latent_feature: int,
            dim_output_time_frame: int,
            dim_output_sequence: int,
            dim_output_feature: int,

            num_hidden_layers: int,
    ) -> None:
        super().__init__(
            name="Gate Recurrent Unit",
            short_name="GRU",

            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_time_frame=dim_input_time_frame,
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_latent_sequence=dim_latent_sequence,
            dim_latent_feature=dim_latent_feature,
            dim_output_time_frame=dim_output_time_frame,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
            sf_model_type=SFGRU
        )
