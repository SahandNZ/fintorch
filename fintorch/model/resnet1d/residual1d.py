from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        self.main_edge = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim)
        )

        self.skip_edge = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
        )

    def forward(self, x):
        main = self.main_edge(x)
        skip = self.skip_edge(x)
        return main + skip
