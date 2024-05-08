import copy
import uuid
from typing import Dict, List

from fintorch.dtype import Order, Position
from fintorch.enum import MarketType, OrderStatus, PositionType, TimeFrame
from fintorch.exchange import Data, Trade, Wallet


class LocalTrade(Trade):
    def __init__(self, market_type: MarketType, wallet: Wallet, data: Data, interval: TimeFrame) -> None:
        super().__init__(exchange_name="Local Exchange", market_type=market_type, interval=interval)
        self.__wallet: Wallet = wallet
        self.__data: Data = data

        self.__symbol_to_leverage: Dict[str, int] = {}
        self.__symbol_to_all_orders_dict: Dict[str, Dict[str, Order]] = {}
        self.__symbol_to_open_orders_dict: Dict[str, Dict[str, Order]] = {}
        self.__symbol_to_position: Dict[str, Position] = {}

        self.__symbol_to_closed_position: Dict[str, List[Position]] = {}

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        self.__handle_open_orders()

    def get_leverage(self, symbol: str) -> int:
        if symbol not in self.__symbol_to_leverage:
            self.__symbol_to_leverage[symbol] = 1

        return self.__symbol_to_leverage[symbol]

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.__symbol_to_leverage[symbol] = leverage

    def get_order(self, symbol: str, order_id: str) -> Order:
        all_orders_dict = self.__get_all_orders_dict(symbol=symbol)
        return all_orders_dict[order_id]

    def __get_open_orders_dict(self, symbol: str) -> Dict[str, Order]:
        if symbol not in self.__symbol_to_open_orders_dict:
            self.__symbol_to_open_orders_dict[symbol] = {}

        return self.__symbol_to_open_orders_dict[symbol]

    def get_open_orders(self, symbol: str) -> List[Order]:
        return list(self.__get_open_orders_dict(symbol=symbol).values())

    def __get_all_orders_dict(self, symbol: str) -> Dict[str, Order]:
        if symbol not in self.__symbol_to_all_orders_dict:
            self.__symbol_to_all_orders_dict[symbol] = {}

        return self.__symbol_to_all_orders_dict[symbol]

    def get_orders_history(self, symbol: str) -> List[Order]:
        return list(self.__get_all_orders_dict(symbol=symbol).values())

    def set_order(self, order: Order) -> Order:
        order.id = str(uuid.uuid4())
        order.timestamp = self.__data.get_current_timestamp()

        # add to all orders dict
        all_orders_dict = self.__get_all_orders_dict(symbol=order.symbol)
        all_orders_dict[order.id] = order

        # add to open orders dict
        open_orders_dict = self.__get_open_orders_dict(symbol=order.symbol)
        open_orders_dict[order.id] = order

        return order

    def cancel_order(self, symbol: str, order_id: str) -> None:
        order = self.get_order(symbol=symbol, order_id=order_id)
        self.__handle_canceled_order(order=order)

    def cancel_all_orders(self, symbol: str) -> None:
        open_orders_dict = self.__get_open_orders_dict(symbol=symbol)
        for order_id in list(open_orders_dict.keys()):
            self.cancel_order(symbol=symbol, order_id=order_id)

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.__symbol_to_position:
            position = Position()
            position.symbol = symbol
            position.type = PositionType.ISOLATED
            position.quantity = 0

            self.__symbol_to_position[symbol] = position

        position = self.__symbol_to_position[symbol]
        position.leverage = self.get_leverage(symbol)
        position.current_price = self.__data.get_current_candle(symbol=symbol, time_frame=self.interval).open

        return position

    def get_positions_history(self, symbol: str) -> List[Position]:
        if symbol not in self.__symbol_to_position:
            self.__symbol_to_closed_position[symbol] = []

        return self.__symbol_to_closed_position[symbol]

    def __handle_open_orders(self):
        for symbol in self.__symbol_to_open_orders_dict.keys():
            for open_order in self.get_open_orders(symbol=symbol):
                current_candle = self.__data.get_current_candle(symbol=symbol, time_frame=self.interval)
                if not open_order.is_active:
                    if not open_order.type.is_stop or current_candle.is_touched(open_order.stop_price):
                        self.__handle_activated_order(order=open_order)

                elif open_order.type.is_market or current_candle.is_touched(open_order.price):
                    self.__handle_filled_order(order=open_order)

    def __handle_activated_order(self, order: Order) -> None:
        current_candle = self.__data.get_current_candle(symbol=order.symbol, time_frame=self.interval)

        # update order properties
        order.activated_timestamp = self.__data.get_current_timestamp()
        order.activated_price = order.stop_price or current_candle.open

        # update balance
        order_price = order.price or current_candle.open
        order_margin = round(order.quantity * order_price / self.get_leverage(symbol=order.symbol), 2)
        balance = self.__wallet.get_balance(market_type=self.market_type, asset=order.base_asset)
        if balance.available < order_margin:
            raise Exception("Insufficient balance (available: {:.2f} required: {:.2f})."
                            .format(balance.available, order_margin))
        else:
            if order.type.is_market:
                balance.total -= order_margin
            else:
                balance.total -= order_margin
                balance.frozen += order_margin

    def __handle_filled_order(self, order: Order) -> None:
        symbol_info = self.__data.get_symbol_info(symbol=order.symbol)
        current_candle = self.__data.get_current_candle(symbol=order.symbol, time_frame=self.interval)

        # update order properties
        order.filled_timestamp = self.__data.get_current_timestamp()
        order.filled_price = order.price or current_candle.open
        order.status = OrderStatus.FILLED

        # remove order it open orders
        del self.__symbol_to_open_orders_dict[order.symbol][order.id]

        # update position and balance
        position = self.get_position(symbol=order.symbol)

        # handle closed position
        if position.is_open and position.side != order.side:
            closed_position = copy.deepcopy(position)
            closed_position.quantity = round(int(closed_position.side) * order.quantity, symbol_info.quantity_precision)
            closed_position.exit_timestamp = self.__data.get_current_timestamp()
            closed_position.exit_price = order.price

            # add closed position to position history
            position_history = self.get_position_history(symbol=position.symbol)
            position_history.append(closed_position)

            # update total balance
            balance = self.__wallet.get_balance(market_type=self.market_type, asset=order.base_asset)
            balance.total += closed_position.realized_profit

        # handle new position
        if not position.is_open:
            position.entry_timestamp = self.__data.get_current_timestamp()

        # update position
        position.quantity = round(position.quantity + int(order.side) * order.quantity, symbol_info.quantity_precision)

    def __handle_canceled_order(self, order: Order) -> None:
        # update order properties
        order.cancel_timestamp = self.__data.get_current_timestamp()
        order.status = OrderStatus.CANCELED

        # remove it from open orders
        del self.__symbol_to_open_orders_dict[order.symbol][order.id]

        # update balance
        order_margin = round(order.quantity * order.price / self.get_leverage(symbol=order.symbol), 2)
        balance = self.__wallet.get_balance(market_type=self.market_type, asset=order.base_asset)
        balance.total += order_margin
        balance.frozen -= order_margin
