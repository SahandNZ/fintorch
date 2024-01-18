from torch import nn

from ._block import Block


class Residual1D(Block):
    def __init__(self, input_dim: int, output_dim: int, batch_norm: bool = False):
        super().__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        main_edge_layers = []
        main_edge_layers.append(nn.Linear(self.input_dim, self.output_dim))
        if batch_norm:
            main_edge_layers.append(nn.BatchNorm1d(self.output_dim))
        main_edge_layers.append(nn.LeakyReLU())
        main_edge_layers.append(nn.Linear(self.output_dim, self.output_dim))
        if batch_norm:
            main_edge_layers.append(nn.BatchNorm1d(self.output_dim))
        self.main_edge = nn.Sequential(*main_edge_layers)

        skip_edge_layers = [nn.Linear(self.input_dim, self.output_dim)]
        if batch_norm:
            skip_edge_layers.append(nn.BatchNorm1d(self.output_dim))
        self.skip_edge = nn.Sequential(*skip_edge_layers)

    def forward(self, x):
        main = self.main_edge(x)
        skip = self.skip_edge(x)
        return main + skip
