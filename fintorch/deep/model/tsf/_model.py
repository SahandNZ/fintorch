import copy
from typing import Type, List

import torch
from torch import nn

from fintorch.utils.hash import static_list_hash
from ..sf import SFModel, SFFeedForward


class TSFModel(SFModel):
    def __init__(
            self,
            name: str,
            short_name: str,
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
            sf_model_type: Type[SFModel]
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers
        )

        self.__dim_input_time_frame: int = dim_input_time_frame
        self.__dim_latent_sequence: int = dim_latent_sequence
        self.__dim_latent_feature: int = dim_input_feature
        self.__dim_output_time_frame: int = dim_output_time_frame

        sf_model: SFModel = sf_model_type(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=nn.Tanh(),

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_latent_sequence,
            dim_output_feature=dim_latent_feature,

            num_hidden_layers=num_hidden_layers
        )

        self.sf_models: List[SFModel] = [copy.deepcopy(sf_model) for _ in range(dim_input_time_frame)]
        self.ff: SFFeedForward = SFFeedForward(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_latent_sequence,
            dim_input_feature=dim_input_time_frame * dim_latent_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_time_frame * dim_output_feature,

            num_hidden_layers=num_hidden_layers
        )

        self.__static_hash: int = static_list_hash([
            super().static_hash,
            self.dim_input_time_frame,
            self.dim_output_time_frame
        ])

    @property
    def dim_input_time_frame(self) -> int:
        return self.__dim_input_time_frame

    @property
    def dim_latent_sequence(self) -> int:
        return self.__dim_latent_sequence

    @property
    def dim_latent_feature(self) -> int:
        return self.__dim_latent_feature

    @property
    def dim_output_time_frame(self) -> int:
        return self.__dim_output_time_frame

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = []
        for index, sf_model in enumerate(self.sf_models):
            time_frame_x = x[range(x.shape[0]), index]
            time_frame_latent = sf_model(time_frame_x)
            latents.append(time_frame_latent)

        latent = torch.stack(latents, dim=1).transpose(1, 2).flatten(start_dim=-2)
        y_hat = self.ff(latent)

        return y_hat

    def to(self, device: torch.device):
        for sf_model in self.sf_models:
            sf_model.to(device)
        self.ff.to(device)

    def reset(self, layers: List[nn.Module] = None):
        for sf_model in self.sf_models:
            sf_model.reset()
