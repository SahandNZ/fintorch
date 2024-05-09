from typing import Generator

from ._clock import Clock
from ._engine import Engine
from ..enum import TimeFrame
from ..exchange import LocalExchange, OnlineExchange
from ..strategy import Strategy


class SimulationEngine(Engine):
    def __init__(
            self,
            online_exchange: OnlineExchange,
            strategy: Strategy,
            interval: TimeFrame,
    ):
        exchange = LocalExchange(online_exchange=online_exchange, interval=interval)
        clock = Clock(online_exchange=online_exchange, strategy=strategy, interval=interval)
        super().__init__(exchange=exchange, strategy=strategy, clock=clock)

    def simulate(self) -> Generator[float, None, None]:
        while True:
            timestamp = self.clock.next()
            if timestamp is not None:
                self.next(timestamp=timestamp)
                yield timestamp
            else:
                break
