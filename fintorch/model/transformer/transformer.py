from torch import nn

from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.model import Model


class Transformer(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, num_head: int, num_layer: int, dropout: float,
                 dim_output: int, activation_fn: nn.Module = None):
        super().__init__(name="Transformer", short_name="TRAN")

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feature, nhead=num_head, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.ff = FeedForward(layers=[dim_sequence * dim_feature, dim_sequence, dim_output], dropout=dropout,
                              activation_fn=activation_fn)

    def forward(self, x):
        f = self.encoder(x)
        y_hat = self.ff(f)
        return y_hat
