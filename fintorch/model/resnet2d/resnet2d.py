from typing import List

from torch import nn

from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.model import Model


class ResNet2D(Model):
    def __init__(self, block: nn.Module, channels: List[int], kernel_size: int, width: int, height: int, dropout: float,
                 dim_output: int, activation_fn: nn.Module = None):
        super().__init__(name="Residual Network 2D", short_name="RN2D")

        dim_reduction = kernel_size ** (2 * (len(channels) - 1))
        dim_input_ff = (width // dim_reduction) * (height // dim_reduction) * channels[-1]

        modules = []
        for index in range(len(channels) - 2):
            modules.append(
                block(in_channels=channels[index], out_channels=channels[index + 1], kernel_size=kernel_size))
            modules.append(nn.Dropout(dropout))
            modules.append(nn.LeakyReLU())

        modules.append(block(in_channels=channels[-2], out_channels=channels[-1], kernel_size=kernel_size))

        self.conv_net = nn.Sequential(*modules)
        self.ff = FeedForward(layers=[dim_input_ff, dim_input_ff, dim_output], dropout=dropout,
                              activation_fn=activation_fn)

    def forward(self, x):
        f = self.conv_net(x)
        y_hat = self.ff(f)

        return y_hat
