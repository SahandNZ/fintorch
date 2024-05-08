from typing import List

from ..._exchange.market import MarketTrade
from ....api import API
from ....dtype import Order, Position
from ....enum import TimeFrame, MarketType


class OnlineMarketTrade(MarketTrade):
    def __init__(self, api: API, market_type: MarketType):
        super().__init__()
        self.__api: API = api
        self.__market_type: MarketType = market_type

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        raise NotImplementedError()

    def next(self, timestamp: int) -> None:
        raise NotImplementedError()

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

    def set_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError()

    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError()

    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError()

    def get_positions_history(self, symbol: str) -> List[Position]:
        raise NotImplementedError()
