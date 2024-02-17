from abc import ABC, abstractmethod

from .network import Https, Network, Wss
from .. import Trade
from ...dtype import Order
from ...enum import MarketType, TimeFrame


class OnlineTrade(Trade, Network, ABC):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame, https: Https, wss: Wss):
        Trade.__init__(self, exchange_name=exchange_name, market_type=market_type, interval=interval)
        Network.__init__(self, https=https, wss=wss)

    def set_order(self, order: Order) -> Order:
        if order.stop_price is None:
            if order.price is None:
                return self._set_market_order(order=order)
            else:
                return self._set_limit_order(order=order)
        else:
            if order.price is None:
                return self._set_stop_market_order(order=order)
            else:
                return self._set_stop_limit_order(order=order)

    @abstractmethod
    def _set_market_order(self, order: Order) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def _set_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def _set_stop_market_order(self, order: Order) -> Order:
        raise NotImplementedError()

    @abstractmethod
    def _set_stop_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()
