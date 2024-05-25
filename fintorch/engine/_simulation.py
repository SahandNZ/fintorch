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
        clock = Clock(start_date="2021-01-01", interval=interval)
        super().__init__(exchange=exchange, strategy=strategy, clock=clock)

        self.__online_exchange: OnlineExchange = online_exchange

    def simulate(self) -> Generator[int, None, None]:
        symbols = [self.strategy.symbol]
        time_frames = self.strategy.time_frames
        dc = self.__online_exchange.future.data.get_data_collection(symbols=symbols, time_frames=time_frames)
        self.strategy.preprocess(dc=dc)

        while True:
            timestamp = self.clock.next()
            if timestamp is not None:
                self.next(timestamp=timestamp)
                yield timestamp
            else:
                break
