from torch import nn

from ._feed_forward import FeedForward
from ._model import Model


class Transformer(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dim_output: int, dropout: float = 0,
                 activation_fn: nn.Module = None, num_head: int = 4, num_layer: int = 1, num_hidden_layers: int = 4,
                 batch_norm: bool = False):
        super().__init__(
            name="Transformer",
            short_name="TRAN",
            dim_sequence=dim_sequence,
            dim_feature=dim_feature,
            dim_output=dim_output,
            dropout=dropout,
            activation_fn=activation_fn
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feature, nhead=num_head, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.ff = FeedForward(dim_sequence=dim_sequence, dim_feature=dim_feature, dim_output=dim_output,
                              dropout=dropout, activation_fn=activation_fn, num_hidden_layers=num_hidden_layers,
                              batch_norm=batch_norm)

    def forward(self, x):
        f = self.encoder(x)
        y_hat = self.ff(f)
        return y_hat
