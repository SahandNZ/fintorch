import os.path
from abc import ABC
from datetime import datetime
from typing import Tuple, Generator

from matplotlib import pyplot as plt

from ._clock import Clock
from ..dtype import DataCollection
from ..exchange import Exchange
from ..settings import FINTORCH_DATA_DIR
from ..strategy import Strategy
from ..utils.plot import draw_position
from ..utils.timestamp import floor_timestamp, ceil_timestamp


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

    def next(self, timestamp: int) -> None:
        self.exchange.next(timestamp=timestamp)
        if 0 == self.exchange.future.data.timestamp % self.strategy.time_frame:
            self.strategy.next(timestamp=timestamp)
            self.strategy.on_new_candle()

    def close(self):
        self.strategy.close()
        self.exchange.close(directory=self.directory)
        self.clock.close(directory=self.directory)

    def draw_positions_ohlcv_plot(
            self,
            dc: DataCollection,
            look_back: int = 100,
            look_ahead: int = 100,
    ) -> Generator[Tuple[plt.Figure, plt.Axes], None, None]:
        symbol = self.strategy.symbol
        time_frame = self.clock.interval

        for position in self.exchange.future.trade.get_positions_history(symbol=symbol):
            start_timestamp = position.entry_timestamp - look_back * int(time_frame)
            stop_timestamp = position.exit_timestamp + look_ahead * int(time_frame)

            start_timestamp = floor_timestamp(start_timestamp, time_frame)
            stop_timestamp = ceil_timestamp(stop_timestamp, time_frame)

            start_date = datetime.fromtimestamp(start_timestamp)
            stop_date = datetime.fromtimestamp(stop_timestamp)

            # draw candlestick, labels and predictions plot
            fig, ax, df = self.strategy.module.dataset.label_transform.draw_ohlc_plot(
                dc=dc,
                start_date=start_date,
                stop_date=stop_date
            )

            draw_position(ax=ax, df=df, position=position)
            fig.show()
            break

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return f"Simulation {str(self.strategy)}"
