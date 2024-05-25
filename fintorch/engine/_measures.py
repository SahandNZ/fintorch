import copy
import math
from typing import List, Union

import numpy as np
import pandas as pd

from . import Engine
from ..dtype import Position
from ..enum import TimeFrame
from ..exchange import Exchange
from ..strategy import Strategy
from ..utils.timestamp import ceil_timestamp


class Measures:
    def __init__(
            self,
            engine: Engine,
            positions: List[Position],
            leverage: int,
            initial_capital: int,
            margin_assignment_method: str,
            fee_percentage: float = 0.1,
            risk_free_percentage: float = 4,
    ):
        self.engine: Engine = engine
        self.positions: List[Position] = copy.deepcopy(positions)
        self.leverage: int = leverage
        self.initial_capital: int = initial_capital
        self.margin_assignment_method: str = margin_assignment_method
        self.fee_percentage: float = fee_percentage
        self.risk_free_percentage: float = risk_free_percentage

        self.exchange: Exchange = engine.exchange
        self.strategy: Strategy = engine.strategy

        self.symbol = self.strategy.symbol
        self.time_frame = self.strategy.time_frame
        self.symbol_info = self.exchange.future.data.get_symbol_info(symbol=self.strategy.symbol)
        self.candles_df = self.exchange.future.data.get_candles_dataframe(
            symbol=self.symbol,
            time_frame=self.engine.clock.interval
        )

        self.__daily_profits: Union[np.ndarray, None] = None

        self.__assign_quantity()
        self.__calculate_daily_profits()

    @property
    def days(self) -> float:
        first_entry_timestamp = self.positions[0].entry_timestamp
        last_exit_timestamp = self.positions[-1].exit_timestamp or self.positions[-1].current_timestamp
        total_duration = last_exit_timestamp - first_entry_timestamp
        return np.round(total_duration / TimeFrame.DAY1)

    @property
    def bars(self) -> np.ndarray:
        return np.round(np.array([item.duration for item in self.positions]) / self.strategy.time_frame)

    @property
    def paid_fees(self) -> np.ndarray:
        return np.array([p.paid_fee(fee_percentage=self.fee_percentage) for p in self.positions])

    @property
    def profits(self) -> np.ndarray:
        profits = np.array([p.profit for p in self.positions])
        return profits - self.paid_fees

    @property
    def daily_profits(self) -> np.ndarray:
        return self.__daily_profits

    @property
    def profit_percentages(self) -> np.ndarray:
        return np.round(self.profits / self.initial_capital * 100, 2)

    @property
    def daily_profit_percentages(self) -> np.ndarray:
        return np.round(self.daily_profits / self.initial_capital * 100, 2)

    @property
    def average_bars_in_trade(self) -> int:
        return np.round(self.bars.mean())

    @property
    def average_bars_in_winning_trade(self) -> int:
        return np.round(self.bars[0 < self.profits].mean())

    @property
    def average_bars_in_losing_trade(self) -> int:
        return np.round(self.bars[self.profits < 0].mean())

    @property
    def net_profit(self) -> float:
        return np.round(self.profits.sum(), 2)

    @property
    def commission_paid(self) -> float:
        return np.round(self.paid_fees.sum(), 2)

    @property
    def gross_profit(self) -> float:
        return np.round(self.profits[0 < self.profits].sum(), 2)

    @property
    def gross_loss(self) -> float:
        return np.abs(np.round(self.profits[self.profits < 0].sum(), 2))

    @property
    def profit_factor(self) -> float:
        return np.round(self.gross_profit / self.gross_loss, 2)

    @property
    def maximum_run_up(self) -> float:
        equity, minimum_equity = self.initial_capital, self.initial_capital
        maximum_run_up = 0
        for position in self.positions:
            run_up = round(equity - minimum_equity + position.run_up, 2)
            maximum_run_up = max(maximum_run_up, run_up)

            equity += position.profit
            minimum_equity = min(minimum_equity, equity)

        return maximum_run_up

    @property
    def maximum_draw_down(self) -> float:
        equity, maximum_equity, maximum_equity_timestamp = self.initial_capital, self.initial_capital, 0
        maximum_draw_down, maximum_draw_down_duration = 0, 0
        for position in self.positions:
            draw_down = round(maximum_equity - equity + abs(position.draw_down), 2)
            maximum_draw_down = max(maximum_draw_down, draw_down)

            position_exit_timestamp = position.exit_timestamp or position.current_timestamp
            draw_down_duration = (position_exit_timestamp - maximum_equity_timestamp) / self.strategy.time_frame
            maximum_draw_down_duration = max(maximum_draw_down_duration, draw_down_duration)

            equity = round(equity + position.profit, 2)
            maximum_equity = max(maximum_equity, equity)
            if equity == maximum_equity:
                maximum_equity_timestamp = position.entry_timestamp

        return maximum_draw_down

    @property
    def buy_and_hold(self) -> float:
        quantity_round_factor = 10 ** self.symbol_info.quantity_precision

        first_timestamp = self.positions[0].entry_timestamp
        last_timestamp = self.positions[-1].exit_timestamp or self.positions[-1].current_timestamp

        entry_price = self.candles_df.loc[first_timestamp].open
        exit_price = self.candles_df.loc[last_timestamp].close

        quantity = self.initial_capital / entry_price
        quantity = math.floor(quantity * quantity_round_factor) / quantity_round_factor
        dust = round(self.initial_capital - quantity * entry_price, 2)
        entry_fee = round(quantity * entry_price * self.fee_percentage / 100, 2)
        exit_fee = round(quantity * exit_price * self.fee_percentage / 100, 2)
        paid_fees = entry_fee + exit_fee

        buy_and_hold = exit_price * quantity - entry_price * quantity + dust - paid_fees
        buy_and_hold = np.round(buy_and_hold, 2)

        return buy_and_hold

    @property
    def sharpe_ratio(self) -> float:
        daily_risk_free_percentage = self.risk_free_percentage / 365
        mean_daily_profit_percentage = self.daily_profit_percentages.mean()
        numerator = mean_daily_profit_percentage - daily_risk_free_percentage
        denominator = self.daily_profit_percentages.std()
        return round(numerator / denominator * np.sqrt(365), 2)

    @property
    def sortino_ratio(self) -> float:
        daily_risk_free_percentage = self.risk_free_percentage / 365
        mean_daily_profit_percentage = self.daily_profit_percentages.mean()
        numerator = mean_daily_profit_percentage - daily_risk_free_percentage
        denominator = self.daily_profit_percentages[self.daily_profit_percentages <= 0].std()
        return round(numerator / denominator * np.sqrt(365), 2)

    @property
    def number_of_total_trades(self) -> int:
        return len(self.positions)

    @property
    def number_of_winning_trades(self) -> int:
        return (0 <= self.profits).sum()

    @property
    def number_of_losing_trades(self) -> int:
        return (self.profits < 0).sum()

    @property
    def winning_ratio(self) -> float:
        return np.round(self.number_of_winning_trades / self.number_of_total_trades * 100, 2)

    @property
    def average_trades(self) -> float:
        return np.round(self.profits.mean(), 2)

    @property
    def average_winning_trades(self) -> float:
        return np.round(self.profits[0 < self.profits].mean(), 2)

    @property
    def average_losing_trades(self) -> float:
        return np.round(np.abs(self.profits[self.profits < 0].mean()), 2)

    @property
    def ratio_average_win_to_average_loss(self) -> float:
        return np.round(self.average_winning_trades / self.average_losing_trades)

    @property
    def largest_winning_trade(self) -> float:
        return np.round(self.profits[0 < self.profits].max(), 2)

    @property
    def largest_losing_trade(self) -> float:
        return np.round(np.abs(self.profits[self.profits < 0]).max(), 2)

    @property
    def equity_series(self) -> List[float]:
        return list(self.initial_capital + np.cumsum(self.profits))

    @property
    def buy_and_hold_series(self) -> List[float]:
        return list(self.initial_capital + np.cumsum(self.buy_and_hold))

    @property
    def draw_down_series(self) -> List[float]:
        equity, maximum_equity = self.initial_capital, self.initial_capital
        series = []
        for position in self.positions:
            draw_down = round(maximum_equity - equity + abs(position.draw_down), 2)
            equity = round(equity + position.profit, 2)
            maximum_equity = max(maximum_equity, equity)

            series.append(draw_down)

        return series

    def __assign_quantity(self):
        quantity_round_factor = 10 ** self.symbol_info.quantity_precision

        equities = [self.initial_capital]
        for position in self.positions:
            position.leverage = self.leverage

            equity = equities[-1] if "compound" == self.margin_assignment_method else self.initial_capital
            equity = max(equity, 0)
            quantity = equity / position.entry_price * position.entry_percentage / 100 * position.leverage
            position.quantity = math.floor(quantity * quantity_round_factor) / quantity_round_factor

            profit = position.profit
            paid_fee = position.paid_fee(fee_percentage=self.fee_percentage)
            equities.append(equities[-1] + profit - paid_fee)

    def __calculate_daily_profits(self):
        df = self.candles_df.copy()
        df["roc"] = df.close / df.open - 1

        profits_dict = {}
        for position in self.positions:
            entry_timestamp = position.entry_timestamp
            exit_timestamp = position.exit_timestamp or position.current_timestamp
            timestamps = list(np.arange(entry_timestamp, exit_timestamp, float(self.engine.clock.interval)))
            for timestamp in timestamps:
                profit = round(df.loc[timestamp].roc * position.side * position.margin, 2)
                profits_dict[timestamp] = profit

        df["profit"] = [profits_dict.get(ts, 0) for ts in df.index]

        # resampling daily profits
        window_size = int(int(TimeFrame.DAY1) // int(self.engine.clock.interval))

        first_timestamp = self.positions[0].entry_timestamp
        first_timestamp = ceil_timestamp(timestamp=first_timestamp, time_frame=TimeFrame.DAY1)
        last_timestamp = self.positions[-1].exit_timestamp or self.positions[-1].current_timestamp
        last_timestamp = ceil_timestamp(timestamp=last_timestamp, time_frame=TimeFrame.DAY1)

        forward_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        ddf = df[first_timestamp <= df.index]
        ddf = ddf[ddf.index <= last_timestamp]
        ddf["profit"] = ddf.profit.rolling(window=forward_indexer, min_periods=1, step=window_size).sum()
        ddf = ddf.dropna()

        self.__daily_profits = ddf.profit.to_numpy()

    def __str__(self) -> str:
        return (
            "Measures:\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"

            "\t- {:<64}{}\n"
            "\t- {:<64}{}\n\n"
        ).format(
            "Days", self.days,
            "Average bars in trade", self.average_bars_in_trade,
            "Average bars in winning trade", self.average_bars_in_winning_trade,
            "Average bars in losing trade", self.average_bars_in_losing_trade,

            "Net profit", self.net_profit,
            "Commission paid", self.commission_paid,

            "Gross profit", self.gross_profit,
            "Gross loss", self.gross_loss,
            "Profit factor", self.profit_factor,

            "Max run up", self.maximum_run_up,
            "Max draw down", self.maximum_draw_down,

            "Buy and hold", self.buy_and_hold,

            "Sharpe ratio", self.sharpe_ratio,
            "Sortino ratio", self.sortino_ratio,

            "Number of total trades", self.number_of_total_trades,
            "Number of winning trades", self.number_of_winning_trades,
            "Number of losing trades", self.number_of_losing_trades,
            "winning ratio", self.winning_ratio,

            "Average trades", self.average_trades,
            "Average winning trades", self.average_winning_trades,
            "Average losing trades", self.average_losing_trades,
            "Average win to average loss ratio", self.ratio_average_win_to_average_loss,

            "Largest winning trade", self.largest_winning_trade,
            "Largest losing trade", self.largest_losing_trade,
        )
