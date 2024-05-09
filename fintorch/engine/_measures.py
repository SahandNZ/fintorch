import copy
import math
from typing import List

import numpy as np

from . import Engine
from ..dtype import Position
from ..enum import TimeFrame
from ..exchange import Exchange
from ..strategy import Strategy
from ..utils.timestamp import floor_timestamp, ceil_timestamp


class Measures:
    def __init__(
            self,
            engine: Engine,
            positions: List[Position],
            leverage: int,
            initial_capital: int,
            margin_assignment_method: str,
            fee_percentage: float = 0.1,
            risk_free_rate: float = 0.04,
    ):
        self.engine: Engine = engine
        self.positions: List[Position] = copy.deepcopy(positions)
        self.leverage: int = leverage
        self.initial_capital: int = initial_capital
        self.margin_assignment_method: str = margin_assignment_method
        self.risk_free_rate: float = risk_free_rate
        self.fee_percentage: float = fee_percentage

        self.exchange: Exchange = engine.exchange
        self.strategy: Strategy = engine.strategy

        self.symbol = self.strategy.symbol
        self.time_frame = self.strategy.time_frame
        self.symbol_info = self.exchange.future.data.get_symbol_info(symbol=self.strategy.symbol)
        self.daily_candles_df = self.exchange.future.data.get_candles_dataframe(
            symbol=self.symbol,
            time_frame=self.time_frame
        )

        self._assign_quantity()

    @property
    def days(self) -> int:
        first_entry_timestamp = self.positions[0].entry_timestamp
        last_exit_timestamp = self.positions[-1].exit_timestamp or self.positions[-1].current_timestamp
        total_duration = last_exit_timestamp - first_entry_timestamp
        return round(total_duration / TimeFrame.DAY1)

    @property
    def bars(self) -> np.ndarray:
        return np.array([round(position.duration / self.strategy.time_frame) for position in self.positions])

    @property
    def paid_fees(self) -> np.ndarray:
        return np.array([p.paid_fee(fee_percentage=self.fee_percentage) for p in self.positions])

    @property
    def profits(self) -> np.ndarray:
        return np.array([p.profit - p.paid_fee(fee_percentage=self.fee_percentage) for p in self.positions])

    @property
    def daily_profits(self) -> np.ndarray:
        df = self.daily_candles_df.copy()
        df["roc"] = df.close / df.open - 1

        daily_profits = {}
        for position in self.positions:
            position_exit_timestamp = position.exit_timestamp or position.current_timestamp
            entry_timestamp = floor_timestamp(timestamp=position.entry_timestamp, time_frame=TimeFrame.DAY1)
            exit_timestamp = floor_timestamp(timestamp=position_exit_timestamp, time_frame=TimeFrame.DAY1)
            position_timestamp = list(np.arange(entry_timestamp, exit_timestamp, float(self.strategy.time_frame)))
            for timestamp in position_timestamp:
                daily_profit = df.roc.loc[timestamp] * position.side * position.margin
                daily_profits[timestamp] = daily_profit

        # set zero daily profit for missing timestamps
        first_position_timestamp = self.positions[0].entry_timestamp
        last_position_timestamp = self.positions[-1].exit_timestamp or self.positions[-1].current_timestamp
        first_day_timestamp = floor_timestamp(timestamp=first_position_timestamp, time_frame=TimeFrame.DAY1)
        last_day_timestamp = ceil_timestamp(timestamp=last_position_timestamp, time_frame=TimeFrame.DAY1)
        timestamps = list(np.arange(first_day_timestamp, last_day_timestamp, float(self.strategy.time_frame)))
        for timestamp in timestamps:
            daily_profits.setdefault(timestamp, 0)

        return np.array(list(daily_profits.values()))

    @property
    def profit_percentages(self) -> np.ndarray:
        return np.array([p.profit / self.initial_capital * 100 for p in self.positions])

    @property
    def daily_profit_percentages(self):
        return np.array([daily_profit / self.initial_capital * 100 for daily_profit in self.daily_profits])

    @property
    def average_bars_in_trade(self) -> int:
        return round(self.bars.mean())

    @property
    def average_bars_in_winning_trade(self) -> int:
        return round(self.bars[0 < self.profits].mean())

    @property
    def average_bars_in_losing_trade(self) -> int:
        return round(self.bars[self.profits < 0].mean())

    @property
    def net_profit(self) -> float:
        return round(self.profits.sum(), 2)

    @property
    def commission_paid(self) -> float:
        return round(self.paid_fees.sum(), 2)

    @property
    def gross_profit(self) -> float:
        return round(self.profits[0 < self.profits].sum(), 2)

    @property
    def gross_loss(self) -> float:
        return abs(round(self.profits[self.profits < 0].sum(), 2))

    @property
    def profit_factor(self) -> float:
        return round(self.gross_profit / self.gross_loss, 2)

    @property
    def maximum_run_up(self) -> float:
        equity, minimum_equity, maximum_run_up = self.initial_capital, self.initial_capital, 0
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
            draw_down = round(maximum_equity - equity - position.draw_down, 2)
            maximum_draw_down = max(maximum_draw_down, draw_down)

            position_exit_timestamp = position.exit_timestamp or position.current_timestamp
            draw_down_duration = (position_exit_timestamp - maximum_equity_timestamp) / self.strategy.time_frame
            maximum_draw_down_duration = max(maximum_draw_down_duration, draw_down_duration)

            equity = round(equity + position.profit, 2)
            maximum_equity = max(maximum_equity, equity)
            if equity == maximum_equity:
                maximum_equity_timestamp = position.entry_timestamp

        return maximum_equity

    @property
    def buy_and_hold(self) -> float:
        first_entry_price = self.positions[0].entry_price
        last_exit_price = self.positions[-1].exit_price or self.positions[-1].current_price
        buy_and_hold_percentage = (last_exit_price / first_entry_price - 1) * self.leverage
        return round(buy_and_hold_percentage * self.initial_capital, 2)

    @property
    def sharpe_ratio(self) -> float:
        daily_risk_free_percentage = self.risk_free_rate * 100 / 365
        mean_daily_profit_percentage = self.daily_profit_percentages.mean()
        numerator = mean_daily_profit_percentage - daily_risk_free_percentage
        denominator = self.daily_profit_percentages.std()
        return round(numerator / denominator * np.sqrt(365), 4)

    @property
    def sortino_ratio(self) -> float:
        daily_risk_free_percentage = self.risk_free_rate * 100 / 365
        mean_daily_profit_percentage = self.daily_profit_percentages.mean()
        numerator = mean_daily_profit_percentage - daily_risk_free_percentage
        denominator = self.daily_profit_percentages[self.daily_profit_percentages <= 0].std()
        return round(numerator / denominator * np.sqrt(365), 4)

    @property
    def total_closed_trades(self) -> int:
        return len(self.positions)

    @property
    def number_of_winning_trades(self) -> int:
        return (0 < self.profits).sum()

    @property
    def number_of_losing_trades(self) -> int:
        return (self.profits < 0).sum()

    @property
    def winning_ratio(self) -> float:
        return round(self.number_of_winning_trades / self.total_closed_trades * 100, 2)

    @property
    def average_trades(self) -> float:
        return round(self.profits.mean(), 2)

    @property
    def average_winning_trades(self) -> float:
        return round(self.profits[0 < self.profits].mean(), 2)

    @property
    def average_losing_trades(self) -> float:
        return round(np.abs(self.profits[self.profits < 0].mean()), 2)

    @property
    def ratio_average_win_to_average_loss(self) -> float:
        return round(self.average_winning_trades / self.average_losing_trades)

    @property
    def largest_winning_trade(self) -> float:
        return round(self.profits[0 < self.profits].max(), 2)

    @property
    def largest_losing_trade(self) -> float:
        return np.abs(self.profits[self.profits < 0]).max()

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
            draw_down = round(maximum_equity - equity - position.draw_down, 2)
            equity = round(equity + position.profit, 2)
            maximum_equity = max(maximum_equity, equity)

            series.append(draw_down)

        return series

    def _assign_quantity(self):
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

            "Total closed trades", self.total_closed_trades,
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
