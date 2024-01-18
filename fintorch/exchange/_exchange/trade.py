from abc import abstractmethod
from typing import List

from .https import Https
from .network import Network
from .wss import Wss
from ...dtype import Balance, Order, Position
from ...enum import OrderSide


class Trade(Network):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https, wss)

    @abstractmethod
    def get_balance(self) -> Balance:
        raise NotImplementedError()

    @abstractmethod
    def get_leverage(self, symbol: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def get_open_orders(self, symbol: str) -> List[Order]:
        raise NotImplementedError()

    @abstractmethod
    def set_market_order(self, symbol: str, side: OrderSide, volume: float) -> str:
        raise NotImplementedError()

    @abstractmethod
    def set_limit_order(self, symbol: str, side: OrderSide, volume: float, price: float) -> str:
        raise NotImplementedError()

    @abstractmethod
    def set_stop_market_order(self, symbol: str, side: OrderSide, volume: float, stop_price: float) -> str:
        raise NotImplementedError()

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_open_position(self, symbol: str) -> Position:
        raise NotImplementedError()
