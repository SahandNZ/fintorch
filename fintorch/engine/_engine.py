from abc import ABC

from ._clock import Clock
from ..exchange import Exchange
from ..strategy import Strategy


class Engine(ABC):
    def __init__(self, exchange: Exchange, clock: Clock):
        self.__exchange: Exchange = exchange
        self.__clock: Clock = clock

    @property
    def _exchange(self) -> Exchange:
        return self.__exchange

    @property
    def _clock(self) -> Clock:
        return self.__clock

    def _prepare(self, strategy: Strategy):
        self._exchange.prepare(symbols=[strategy.symbol], time_frames=[strategy.time_frame])
        strategy.prepare(exchange=self._exchange)

    def _next(self, strategy: Strategy, timestamp):
        self._exchange.next(timestamp=timestamp)
        if 0 == self._exchange.future.data.timestamp % strategy.time_frame:
            strategy.next(exchange=self._exchange)
