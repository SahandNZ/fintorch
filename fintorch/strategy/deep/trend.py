from fintorch.deep.module import Module
from .deep import DeepStrategy

from ...exchange import Exchange


class DeepTrendStrategy(DeepStrategy):
    def __init__(self, module: Module):
        super().__init__(name="Deep Trend Strategy", short_name="Deep Trend", module=module)

    def _next(self, exchange: Exchange) -> None:
        pass
