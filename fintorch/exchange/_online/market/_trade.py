from typing import List, Union

from ..._exchange.market import MarketTrade
from ....api import API
from ....dtype import Order, Position
from ....enum import MarketType, OrderSide


class OnlineMarketTrade(MarketTrade):

    def __init__(self, api: API, market_type: MarketType):
        super().__init__(market_type=market_type)
        self.__api: API = api

    def get_leverage(self, symbol: str) -> int:
        raise NotImplementedError()

    def set_leverage(self, symbol: str, leverage: int) -> None:
        raise NotImplementedError()

    def get_order(self, symbol: str, order_id: str) -> Order:
        raise NotImplementedError()

    def get_open_orders(self, symbol: str) -> List[Order]:
        raise NotImplementedError()

    def get_orders_history(self, symbol: str) -> List[Order]:
        raise NotImplementedError()

    def get_exit_orders(self, position: Position) -> List[Order]:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def set_entry_order(
            self,
            symbol: str,
            side: OrderSide,
            percentage: float,
            price: Union[float, None] = None,
            stop_price: Union[float, None] = None,
            comment: Union[str, None] = None
    ) -> Order:
        raise NotImplementedError()

    def set_exit_order(
            self,
            position: Position,
            percentage: float,
            price: Union[float, None] = None,
            stop_price: Union[float, None] = None,
            comment: Union[str, None] = None
    ) -> Order:
        raise NotImplementedError()

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError()

    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError()

    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError()

    def get_positions_history(self, symbol: str) -> List[Position]:
        raise NotImplementedError()
