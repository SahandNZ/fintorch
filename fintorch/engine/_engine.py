import os.path
from abc import ABC
from datetime import datetime
from typing import Tuple, Generator, Union

import pandas as pd
from matplotlib import pyplot as plt

from ._clock import Clock
from ._measures import Measures
from ..dtype import DataCollection
from ..exchange import Exchange
from ..settings import FINTORCH_ENGINE_DIR
from ..strategy import Strategy
from ..utils.plot import draw_position_and_orders
from ..utils.timestamp import to_timestamp


class Engine(ABC):
    def __init__(self, exchange: Exchange, strategy: Strategy, clock: Clock):
        self.__exchange: Exchange = exchange
        self.__strategy: Strategy = strategy
        self.__clock: Clock = clock

        self.__directory = os.path.join(FINTORCH_ENGINE_DIR, str(self.strategy.static_hash))

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

    def calculate_measures(
        self,
        start_date: Union[str, datetime],
        stop_date: Union[str, datetime],
        leverage: int = 1,
        initial_capital: int = 1000,
        margin_assignment_method: str ="cumulative"
    ) -> Measures:
        start_timestamp = to_timestamp(date=start_date)
        stop_timestamp = to_timestamp(date=stop_date)

        positions = self.exchange.future.trade.get_positions_history(symbol=self.strategy.symbol)
        filtered_position = []
        for position in positions:
            if start_timestamp <= position.entry_timestamp <= stop_timestamp:
                filtered_position.append(position)

        measures = Measures(
            interval=self.clock.interval,
            strategy=self.strategy,
            exchange=self.exchange,
            positions=filtered_position,
            leverage=leverage,
            initial_capital=initial_capital,
            margin_assignment_method=margin_assignment_method
        )

        return measures

    def draw_positions_plot(
            self,
            dc: DataCollection,
            start_date: Union[str, datetime],
            stop_date: Union[str, datetime],
            look_back: int = 50,
            look_ahead: int = 50,
    ) -> Generator[Tuple[plt.Figure, plt.Axes, pd.DataFrame], None, None]:
        start_timestamp = to_timestamp(date=start_date)
        stop_timestamp = to_timestamp(date=stop_date)

        positions = self.exchange.future.trade.get_positions_history(symbol=self.strategy.symbol)
        for position in positions:
            entry_timestamp = position.entry_timestamp
            exit_timestamp = position.exit_timestamp or position.current_timestamp

            if start_timestamp <= entry_timestamp and exit_timestamp < stop_timestamp:
                exit_orders = self.exchange.future.trade.get_exit_orders(position=position)

                # draw candlestick, indicators, and position plot
                fig, ohlc_ax, df = self.strategy.draw_candlestick_and_indicators_plot(
                    dc=dc,
                    position=position,
                    interval=self.clock.interval,
                    look_back=look_back,
                    look_ahead=look_ahead
                )

                # draw position and exit orders
                draw_position_and_orders(ohlc_ax=ohlc_ax, df=df, position=position, exit_orders=exit_orders)

                yield fig, ohlc_ax, df

    def show_positions_plot(
            self,
            dc: DataCollection,
            start_date: Union[str, datetime],
            stop_date: Union[str, datetime],
            look_back: int = 50,
            look_ahead: int = 50,
    ) -> None:
        fig_generator = self.draw_positions_plot(
            dc=dc,
            start_date=start_date,
            stop_date=stop_date,
            look_back=look_back,
            look_ahead=look_ahead
        )
        for _, _, _ in fig_generator:
            plt.show()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return f"Simulation {str(self.strategy)}"
