import torch
from torch import nn

from ._feed_forward import FeedForward
from ._model import Model


class Hybrid(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, num_hidden_layers: int = 2,
                 batch_norm: bool = True, dropout: float = 0.5, activation_fn: nn.Module = None):
        super().__init__(
            name="Hybrid",
            short_name="HYB",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn
        )
        self.q_linear = nn.Linear(dim_feature, dim_feature)
        self.k_linear = nn.Linear(dim_feature, dim_feature)
        self.v_linear = nn.Linear(dim_feature, dim_feature)

        self.ff = FeedForward(
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
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
