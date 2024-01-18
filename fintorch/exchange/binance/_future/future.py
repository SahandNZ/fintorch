from typing import Dict

from .https import BinanceFutureHttps
from .market import BinanceFutureMarket
from .trade import BinanceFutureTrade
from .wss import BinanceFutureWss
from ..._exchange import Future


class BinanceFuture(Future):
    def __init__(self, key: str = None, secret_key: str = None, proxies: Dict[str, str] = None):
        https = BinanceFutureHttps(key=key, secret_key=secret_key, proxies=proxies)
        # wss = BinanceFutureWss(key=key, secret_key=secret_key, proxies=proxies)
        wss = None

        market = BinanceFutureMarket(https=https, wss=wss)
        trade = BinanceFutureTrade(https=https, wss=wss)

        super().__init__(market=market, trade=trade)
