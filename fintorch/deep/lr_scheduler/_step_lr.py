import torch

from ._lr_scheduler import LRScheduler


class StepLR(LRScheduler):
    def __init__(self, **kwargs):
        super().__init__(lr_scheduler=torch.optim.lr_scheduler.StepLR, **kwargs)
