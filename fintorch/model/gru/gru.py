import torch.nn as nn

from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.model import Model


class GRU(Model):
    def __init__(self, dim_sequence_length: int, dim_feature: int, hidden_size: int, num_layer: int, dropout: float,
                 dim_output: int,
                 activation_fn: nn.Module, auto_cuda: bool = True):
        super().__init__(auto_cuda)
        self.gru = nn.GRU(input_size=dim_feature, hidden_size=hidden_size, num_layers=num_layer, batch_first=True,
                          dropout=dropout)
        self.ff = FeedForward(layers=[dim_sequence_length * hidden_size, dim_sequence_length, dim_output],
                              dropout=dropout, activation_fn=activation_fn)

    def forward(self, x):
        x, _ = self.gru(x)
        y_hat = self.ff(x)
        return y_hat
