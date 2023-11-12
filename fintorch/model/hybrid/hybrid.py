import torch
from torch import nn

from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.model import Model


class Hybrid(Model):
    def __init__(self, dim_sequence: int, dim_feature: int, dropout: float, dim_output: int,
                 activation_fn: nn.Module = None):
        super().__init__(name="Hybrid", short_name="HYB")
        self.q_linear = nn.Linear(dim_feature, dim_feature)
        self.k_linear = nn.Linear(dim_feature, dim_feature)
        self.v_linear = nn.Linear(dim_feature, dim_feature)

        self.ff = FeedForward(layers=[dim_sequence * dim_feature, dim_sequence, dim_output], dropout=dropout,
                              activation_fn=activation_fn)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        s = nn.functional.softmax(torch.einsum('bsf,bsf->bs', q, k), dim=-1).unsqueeze(-1)
        f = v * s
        y_hat = self.ff(f)

        return y_hat
