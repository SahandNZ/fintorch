from typing import Dict, List

from ._network.https import BinanceFutureHttps
from ..._online import OnlineTrade
from ....dtype import Order, Position
from ....enum import MarketType, TimeFrame


class BinanceFutureTrade(OnlineTrade):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame, key: str = None,
                 secret_key: str = None, proxies: Dict = None):
        https = BinanceFutureHttps(key=key, secret_key=secret_key, proxies=proxies)
        super().__init__(exchange_name=exchange_name, market_type=market_type, interval=interval, https=https, wss=None)

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

    def _set_market_order(self, order: Order) -> str:
        raise NotImplementedError()

    def _set_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def _set_stop_market_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def _set_stop_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError()

    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError()

    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError()

    def get_positions_history(self, symbol: str) -> List[Position]:
        raise NotImplementedError()
