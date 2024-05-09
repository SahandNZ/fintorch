import copy
import logging
import math
import uuid
from typing import Dict, List, Union

from ._data import LocalMarketData
from ..._exchange.market import MarketTrade
from ....dtype import Order, Position
from ....enum import MarketType, OrderStatus, TimeFrame, PositionType, PositionSide, OrderSide

_logger = logging.getLogger("fintorch")


class LocalMarketTrade(MarketTrade):
    def __init__(self, interval: TimeFrame, data: LocalMarketData, market_type: MarketType, ) -> None:
        super().__init__(market_type=market_type)
        self.__interval: TimeFrame = interval
        self.__data: LocalMarketData = data

        self.__symbol_to_leverage_dict: Union[Dict[str, int], None] = None
        self.__symbol_to_all_orders_dict: Union[Dict[str, Dict[str, Order]], None] = None
        self.__symbol_to_open_orders_dict: Union[Dict[str, Dict[str, Order]], None] = None
        self.__symbol_to_all_positions_dict: Union[Dict[str, Dict[str, Position]], None] = None
        self.__symbol_to_open_position_dict: Union[Dict[str, Position], None] = None

        self.__order_logs: List[(str, Order)] = []
        self.__position_logs: List[(str, Position)] = []

    def get_leverage(self, symbol: str) -> int:
        self.__symbol_to_leverage_dict.setdefault(symbol, 1)
        return self.__symbol_to_leverage_dict[symbol]

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.__symbol_to_leverage_dict[symbol] = leverage

    def get_order(self, symbol: str, order_id: str) -> Order:
        self.__symbol_to_all_orders_dict.setdefault(symbol, {})
        return self.__symbol_to_all_orders_dict[symbol][order_id]

    def get_open_orders(self, symbol: str) -> List[Order]:
        self.__symbol_to_open_orders_dict.setdefault(symbol, {})
        return list(self.__symbol_to_open_orders_dict[symbol].values())

    def get_orders_history(self, symbol: str) -> List[Order]:
        self.__symbol_to_all_orders_dict.setdefault(symbol, {})
        return list(self.__symbol_to_all_orders_dict[symbol].values())

    def set_order(
            self,
            symbol: str,
            side: OrderSide,
            percentage: float,
            reduce_only: bool,
            price: Union[float, None] = None,
            stop_price: Union[float, None] = None,
            comment: Union[str, None] = None
    ) -> Order:
        symbol_info = self.__data.get_symbol_info(symbol=symbol)

        order = Order()
        order.symbol = symbol
        order.side = OrderSide(side)
        order.percentage = percentage
        order.reduce_only = reduce_only

        order.price = price
        order.stop_price = stop_price
        order.comment = comment

        # round order properties
        order.percentage = math.floor(order.percentage * 10 ** 2) / 10 ** 2
        if order.price is not None:
            order.price = round(order.price, symbol_info.price_precision)
        if order.stop_price is not None:
            order.stop_price = round(order.stop_price, symbol_info.price_precision)

        # set order post processing properties
        order.id = str(uuid.uuid4())
        order.timestamp = self.timestamp
        order.status = OrderStatus.OPEN

        if order.percentage < 0:
            raise RuntimeError("Percentage of order must be greater than zero.")
        if 100 < order.percentage:
            raise RuntimeError("Percentage of order must not exceed 100.")

        # add to all orders dict add open orders dict
        self.__symbol_to_all_orders_dict.setdefault(order.symbol, {})
        self.__symbol_to_open_orders_dict.setdefault(order.symbol, {})
        self.__symbol_to_all_orders_dict[order.symbol][order.id] = order
        self.__symbol_to_open_orders_dict[order.symbol][order.id] = order

        return order

    def cancel_order(self, symbol: str, order_id: str) -> None:
        order = self.get_order(symbol=symbol, order_id=order_id)
        self.__handle_canceled_order(order=order)

    def cancel_all_orders(self, symbol: str) -> None:
        self.__symbol_to_open_orders_dict.setdefault(symbol, {})
        open_orders_dict = self.__symbol_to_open_orders_dict[symbol].copy()
        for order_id in list(open_orders_dict.keys()):
            self.cancel_order(symbol=symbol, order_id=order_id)

    def get_position(self, symbol: str) -> Position:
        # set default position if it's not exists
        default_position = Position()
        default_position.id = str(uuid.uuid4())
        default_position.symbol = symbol
        default_position.type = PositionType.CROSS
        default_position.entry_percentage = 0
        default_position.exit_percentage = 0
        self.__symbol_to_open_position_dict.setdefault(symbol, default_position)

        # update position properties
        position = self.__symbol_to_open_position_dict[symbol]
        position.leverage = self.get_leverage(symbol=symbol)
        position.current_timestamp = self.timestamp
        position.current_price = self.__data.get_current_candle(symbol=symbol, time_frame=self.__interval).close

        return position

    def get_positions_history(self, symbol: str) -> List[Position]:
        self.__symbol_to_all_positions_dict.setdefault(symbol, {})
        return list(self.__symbol_to_all_positions_dict[symbol].values())

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update({
            "symbol_to_leverage_dict": self.__symbol_to_leverage_dict,
            "symbol_to_all_orders_dict": self.__symbol_to_all_orders_dict,
            "symbol_to_open_orders_dict": self.__symbol_to_open_orders_dict,
            "symbol_to_all_positions_dict": self.__symbol_to_all_positions_dict,
            "symbol_to_open_position_dict": self.__symbol_to_open_position_dict,
        })

        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        self.__symbol_to_leverage_dict = state_dict.get("symbol_to_leverage_dict", {})
        self.__symbol_to_all_orders_dict = state_dict.get("symbol_to_all_orders_dict", {})
        self.__symbol_to_open_orders_dict = state_dict.get("symbol_to_open_orders_dict", {})
        self.__symbol_to_all_positions_dict = state_dict.get("symbol_to_all_positions_dict", {})
        self.__symbol_to_open_position_dict = state_dict.get("symbol_to_open_position_dict", {})

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        self.__order_logs = []
        self.__position_logs = []

        self.__handle_open_orders()
        self.__handle_open_positions()
        self.__handle_logs()

    def __handle_open_orders(self):
        for open_orders_dict in self.__symbol_to_open_orders_dict.values():
            for open_order in open_orders_dict.copy().values():
                if open_order.id in open_orders_dict:
                    candle = self.__data.get_current_candle(symbol=open_order.symbol, time_frame=self.__interval)
                    if not open_order.is_activated:
                        if not open_order.type.is_stop or candle.is_touched(open_order.stop_price):
                            self.__handle_activated_order(order=open_order)

                    if open_order.is_activated:
                        if open_order.type.is_market or candle.is_touched(open_order.price):
                            self.__handle_filled_order(order=open_order)

    def __handle_activated_order(self, order: Order) -> None:
        symbol_info = self.__data.get_symbol_info(symbol=order.symbol)
        current_candle = self.__data.get_current_candle(symbol=order.symbol, time_frame=self.__interval)

        # update order properties
        order.activated_timestamp = self.timestamp
        order.activated_price = round(order.stop_price or current_candle.close, symbol_info.price_precision)

        # TODO order percentage validation

        self.__order_logs.append(("activated", copy.deepcopy(order)))

    def __handle_filled_order(self, order: Order) -> None:
        symbol_info = self.__data.get_symbol_info(symbol=order.symbol)
        current_candle = self.__data.get_current_candle(symbol=order.symbol, time_frame=self.__interval)

        # update order properties
        order.filled_timestamp = self.timestamp
        order.filled_price = round(order.price or current_candle.close, symbol_info.price_precision)
        order.status = OrderStatus.FILLED

        # remove it from open orders dict
        open_orders_dict = self.__symbol_to_open_orders_dict[order.symbol]
        open_orders_dict.pop(order.id)

        self.__order_logs.append(("filled", copy.deepcopy(order)))

        # update position
        self.__handle_position(order=order)

    def __handle_canceled_order(self, order: Order) -> None:
        # update order properties
        order.canceled_timestamp = self.timestamp
        order.status = OrderStatus.CANCELED

        # remove it from open orders dict
        open_orders_dict = self.__symbol_to_open_orders_dict[order.symbol]
        open_orders_dict.pop(order.id)

        self.__order_logs.append(("canceled", copy.deepcopy(order)))

    def __handle_position(self, order: Order):
        symbol_info = self.__data.get_symbol_info(symbol=order.symbol)
        position = self.get_position(symbol=order.symbol)

        if not position.is_open:
            if order.reduce_only:
                raise RuntimeError("There is no open position to reduce its size.")
            else:
                position.side = PositionSide(order.side)
                position.entry_price = round(order.filled_price, symbol_info.price_precision)
                position.entry_timestamp = order.filled_timestamp
                position.entry_percentage = math.floor(order.percentage * 10 ** 2) / 10 ** 2
                position.highest_met_price = position.entry_price
                position.lowest_met_price = position.entry_price

                self.__symbol_to_all_positions_dict.setdefault(position.symbol, {})
                self.__symbol_to_all_positions_dict[position.symbol][position.id] = position

                self.__position_logs.append(("opened", copy.deepcopy(position)))

                self.opened_position_event.trigger(args=(position,))

        else:
            if order.reduce_only:
                exit_price = position.exit_price or 0
                numerator = exit_price * position.exit_percentage + order.filled_price * order.percentage
                denominator = position.exit_percentage + order.percentage
                new_exit_price = numerator / denominator
                new_exit_percentage = position.exit_percentage + order.percentage

                position.exit_price = round(new_exit_price, symbol_info.price_precision)
                position.exit_percentage = math.floor(new_exit_percentage * 10 ** 2) / 10 ** 2

                if position.is_closed:
                    position.exit_timestamp = order.filled_timestamp
                    self.__symbol_to_all_positions_dict.setdefault(position.symbol, {})
                    self.__symbol_to_all_positions_dict[position.symbol][position.id] = position
                    self.__symbol_to_open_position_dict.pop(position.symbol)

                    self.__position_logs.append(("closed", copy.deepcopy(position)))
                    self.closed_position_event.trigger(args=(position,))

                else:
                    self.__position_logs.append(("reduced", copy.deepcopy(position)))

            else:
                numerator = position.entry_price * position.entry_percentage + order.filled_price * order.percentage
                denominator = position.entry_percentage + order.percentage
                new_entry_price = numerator / denominator
                new_percentage = position.entry_percentage + order.percentage

                position.entry_price = round(new_entry_price, symbol_info.price_precision)
                position.entry_percentage = math.floor(new_percentage * 10 ** 2) / 10 ** 2

                self.__position_logs.append(("extended", copy.deepcopy(position)))

    def __handle_open_positions(self) -> None:
        for symbol, position in self.__symbol_to_open_position_dict.items():
            if position.is_open:
                current_candle = self.__data.get_current_candle(symbol=symbol, time_frame=self.__interval)
                position.highest_met_price = max(position.highest_met_price, current_candle.high)
                position.lowest_met_price = min(position.lowest_met_price, current_candle.low)

    def __handle_logs(self) -> None:
        if 0 < len(self.__order_logs) + len(self.__position_logs):
            _logger.debug("")
            _logger.debug(f"{str(self.datetime)}")
            _logger.debug(f"open orders count: {len(self.__symbol_to_open_orders_dict['BTC-USDT'])}")

            for title, order in self.__order_logs:
                self.__order_log_fn(title=title, order=order)
            for title, position in self.__position_logs:
                self.__position_log_fn(title=title, position=position)

    def __order_log_fn(self, title: str, order: Order) -> None:
        log = (
            "order {:<12}  ({:<8} - {:<6} - {:<4} - {:<6} - {:^8} - {:^8}) ({:^8} - {:^8})"
            .format(
                title,
                order.symbol,
                str(order.type),
                str(order.side),
                order.percentage,
                order.price or "nan",
                order.stop_price or "nan",
                order.activated_price or "nan",
                order.filled_price or "nan"
            )
        )
        _logger.debug(log)

    def __position_log_fn(self, title: str, position: Position) -> None:
        log = (
            "position {:<10} ({:<8} - {:<5}) ({} - {:<8} - {:<6}) ({:^8} - {:^6} - {:<4})."
            .format(
                title,
                position.symbol,
                str(position.side),
                position.entry_datetime,
                position.entry_price,
                position.entry_percentage,
                position.exit_price or "nan",
                position.exit_percentage or "nan",
                position.profit_percentage,
            )
        )
        _logger.debug(log)
