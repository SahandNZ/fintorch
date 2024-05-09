import os.path
from abc import ABC

from ._clock import Clock
from ..exchange import Exchange
from ..settings import FINTORCH_DATA_DIR
from ..strategy import Strategy


class Engine(ABC):
    def __init__(self, exchange: Exchange, strategy: Strategy, clock: Clock):
        self.__exchange: Exchange = exchange
        self.__strategy: Strategy = strategy
        self.__clock: Clock = clock

        engine_type = self.__class__.__name__.lower().replace("engine", "")
        self.__directory = os.path.join(FINTORCH_DATA_DIR, "engine", engine_type, str(self.strategy.static_hash))

    @property
    def exchange(self) -> Exchange:
        return self.__exchange

    @property
    def strategy(self) -> Strategy:
        return self.__strategy

    @property
    def clock(self) -> Clock:
        return self.__clock

    @property
    def directory(self) -> str:
        return self.__directory

    def open(self):
        self.strategy.open()
        self.exchange.open(directory=self.directory)
        self.clock.open(directory=self.directory)

        # share properties between exchange and strategy
        self.strategy.exchange = self.exchange
        self.exchange.future.trade.opened_position_event.add(self.strategy.on_opened_position)
        self.exchange.future.trade.closed_position_event.add(self.strategy.on_closed_position)

        return self

    def next(self, timestamp: float) -> None:
        self.exchange.next(timestamp=timestamp)
        if 0 == self.exchange.future.data.timestamp % self.strategy.time_frame:
            self.strategy.on_new_candle()

    def close(self):
        self.strategy.close()
        self.exchange.close(directory=self.directory)
        self.clock.close(directory=self.directory)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return f"Simulation {str(self.strategy)}"
