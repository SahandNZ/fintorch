import torch


class Sample:
    def __init__(self, timestamp: int, feature: torch.Tensor, label: torch.Tensor):
        self.timestamp: int = timestamp
        self.feature: torch.Tensor = feature
        self.label: torch.Tensor = label
