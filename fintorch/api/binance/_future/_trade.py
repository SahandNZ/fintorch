from typing import Dict, List

from .network.https import BinanceFutureHttps
from .network.wss import BinanceFutureWss
from ..._api.market import MarketTradeEndPoints
from ....dtype import Order, Position


class BinanceFutureMarketTradeEndPoints(MarketTradeEndPoints):
    def __init__(self, api_key: str, secret_key: str = None, proxies: Dict = None):
        https = BinanceFutureHttps(api_key=api_key, secret_key=secret_key, proxies=proxies)
        wss = BinanceFutureWss(api_key=api_key, secret_key=secret_key, proxies=proxies)
        super().__init__(https=https, wss=wss)

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

    def set_market_order(self, order: Order) -> str:
        raise NotImplementedError()

    def set_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def set_stop_market_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def set_stop_limit_order(self, order: Order) -> Order:
        raise NotImplementedError()

    def cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError()

    def cancel_all_orders(self, symbol: str) -> None:
        raise NotImplementedError()

    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError()

    def get_positions_history(self, symbol: str) -> List[Position]:
        raise NotImplementedError()
