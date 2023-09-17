from torch import nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.main_edge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_edge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size ** 2, stride=kernel_size ** 2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        main = self.main_edge(x)
        skip = self.skip_edge(x)

        return main + skip
