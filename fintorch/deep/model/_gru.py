import torch.nn as nn

from ._feed_forward import FeedForward
from ._model import Model


class GRU(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, num_hidden_layers: int = 2,
                 batch_norm: bool = True, dropout: float = 0.5, activation_fn: nn.Module = None, dim_hidden: int = 32,
                 num_layers: int = 2):
        super().__init__(
            name="Gated Recurrent Unit",
            short_name="GRU",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn
        )

        self.gru = nn.GRU(
            input_size=dim_feature,
            hidden_size=dim_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.ff = FeedForward(
            dim_sequence=dim_sequence,
            dim_feature=dim_hidden,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn
        )

    def _forward(self, x):
        x, _ = self.gru(x)
        y_hat = self.ff(x)
        return y_hat
