from torch import nn

from ._feed_forward import FeedForward
from ._model import Model


class Transformer(Model):
    def __init__(
            self,
            dim_sequence: int,
            dim_feature: int,
            dim_output: int,
            num_hidden_layers: int = 2,
            dropout: float = 0.5,
            batch_norm: bool = True,
            activation_fn: nn.Module = None,
            num_head: int = 2,
            num_layers: int = 2
    ):
        super().__init__(
            name="Transformer",
            short_name="TRAN",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feature,
            nhead=num_head,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        f = self.encoder(x)
        y_hat = self.ff(f)
        return y_hat
