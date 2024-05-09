import torch
from torch import nn

from ._feed_forward import FeedForward
from ._model import Model


class Hybrid(Model):
    def __init__(
            self,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,
            num_hidden_layers: int,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module = None
    ):
        super().__init__(
            name="Hybrid",
            short_name="HYB",
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn
        )

        self.q_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)
        self.k_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)
        self.v_linear = nn.Linear(self.dim_input_feature, self.dim_input_feature)

        self.ff = FeedForward(
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn
        )

    def _forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        s = nn.functional.softmax(torch.einsum('bsf,bsf->bs', q, k), dim=-1).unsqueeze(-1)
        f = v * s

        y_hat = self.ff(f)

        return y_hat
