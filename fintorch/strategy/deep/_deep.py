from abc import ABC

from .. import Strategy
from ...deep.module import Module


class DeepStrategy(Strategy, ABC):
    def __init__(self, name: str, short_name: str, module: Module):
        super().__init__(name=name, short_name=short_name, symbol=module.symbol, time_frame=module.time_frame)
        self.__module: Module = module

    @property
    def module(self) -> Module:
        return self.__module

    def open(self) -> None:
        self.module.open()

    def close(self) -> None:
        self.module.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
