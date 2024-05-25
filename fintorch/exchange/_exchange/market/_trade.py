from abc import abstractmethod, ABC
from typing import List, Union

from ._element import MarketElement
from ....dtype import Order, Position
from ....enum import MarketType, OrderSide
from ....utils.event import Event


class MarketTrade(MarketElement, ABC):
    def __init__(self, market_type: MarketType):
        super().__init__(market_type=market_type)

        self.__opened_position_event: Event = Event()
        self.__closed_position_event: Event = Event()

    @property
    def opened_position_event(self) -> Event:
        return self.__opened_position_event

    @property
    def closed_position_event(self) -> Event:
        return self.__closed_position_event

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

    @abstractmethod
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

    @abstractmethod
    def set_exit_order(
            self,
            position: Position,
            percentage: float,
            price: Union[float, None] = None,
            stop_price: Union[float, None] = None,
            comment: Union[str, None] = None
    ) -> Order:
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
