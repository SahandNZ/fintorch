import time

import matplotlib.pyplot as plt
from rich.progress import Progress

from ._clock import Clock
from ._engine import Engine
from ..enum import TimeFrame
from ..exchange import LocalExchange, OnlineExchange
from ..strategy import Strategy
from ..utils.plot import draw_candlestick_plot, draw_position


class SimulationEngine(Engine):
    def __init__(self, online_exchange: OnlineExchange, initial_capital: float, interval: TimeFrame, speed: int = None):
        exchange = LocalExchange(
            online_exchange=online_exchange,
            initial_capital=initial_capital,
            interval=interval
        )
        clock = Clock(interval=interval, speed=speed)
        super().__init__(exchange=exchange, clock=clock)

    def backtest(self, strategy: Strategy, start_date: str, stop_date: str = None, progress: Progress = None):
        timestamp_generator = self._clock(start_date=start_date, stop_date=stop_date)
        if progress is not None:
            task = progress.add_task(f"Simulating {str(strategy)} Strategy", total=len(self._clock))

        self._prepare(strategy)
        for timestamp in timestamp_generator:
            self._next(strategy=strategy, timestamp=timestamp)
            if progress is not None:
                progress.update(task_id=task, advance=1)

    def simulate(self, strategy: Strategy, start_date: str, stop_date: str, progress: Progress = None):
        plt.ion()
        fig, ohlcv_ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        self._prepare(strategy)
        for timestamp in self._clock(start_date=start_date, stop_date=stop_date):
            self._next(strategy=strategy, timestamp=timestamp)

            # get candles dataframe
            symbol = strategy.symbol
            time_frame = self._clock.interval
            df = self._exchange.future.data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)[-100:]

            # draw candlestick plot
            ohlcv_ax.cla()
            ohlcv_ax.grid()
            draw_candlestick_plot(ax=ohlcv_ax, df=df)

            # draw all orders
            # for open_order in self.exchange.future.trade.get_orders_history(symbol=symbol):
            #     draw_order(ax=ohlcv_ax, df=df, order=open_order)

            # draw position history
            for position in self._exchange.future.trade.get_positions_history(symbol=symbol):
                draw_position(ax=ohlcv_ax, df=df, position=position)

            # draw open position
            position = self._exchange.future.trade.get_position(symbol=strategy.symbol)
            if position.is_open:
                draw_position(ax=ohlcv_ax, df=df, position=position)

            plt.pause(0.1)
