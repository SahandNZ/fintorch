import torch
from torch import nn

from ._feed_forward import SFFeedForward
from ._model import SFModel


class SFHybrid(SFModel):
    def __init__(
            self,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module,

            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,

            num_hidden_layers: int,
    ):
        super().__init__(
            name="SF Hybrid",
            short_name="SF HYB",
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
        )

        self.q_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)
        self.k_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)
        self.v_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)

        self.ff = SFFeedForward(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
        )

    def _forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        s = nn.functional.softmax(torch.einsum('bsf,bsf->bs', q, k), dim=-1).unsqueeze(-1)
        f = v * s

        y_hat = self.ff(f)

        return y_hat
