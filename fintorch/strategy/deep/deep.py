from abc import ABC
from typing import List

from .. import Strategy
from ...deep.module import Module
from ...enum import TimeFrame


class DeepStrategy(Strategy, ABC):
    def __init__(self, name: str, short_name: str, module: Module):
        super().__init__(name=name, short_name=short_name)
        self.__module: Module = module

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def symbols(self) -> List[str]:
        return [self.module.dataset.label_transform.symbol]

    @property
    def time_frames(self) -> List[TimeFrame]:
        return [self.module.dataset.label_transform.time_frame]
