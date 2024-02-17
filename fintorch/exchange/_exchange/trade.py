from abc import ABC, abstractmethod
from typing import List

from .element import Element
from ...dtype import Order, Position
from ...enum import MarketType, TimeFrame


class Trade(Element, ABC):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame) -> None:
        super().__init__(exchange_name=exchange_name, market_type=market_type, interval=interval)

    @abstractmethod
    def get_leverage(self, symbol: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def get_open_orders(self, symbol: str) -> List[Order]:
        raise NotImplementedError()

    @abstractmethod
    def get_orders_history(self, symbol: str) -> List[Order]:
        raise NotImplementedError()

    @abstractmethod
    def set_order(self, order: Order) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError()

    @abstractmethod
    def get_positions_history(self, symbol: str) -> List[Position]:
        raise NotImplementedError()
