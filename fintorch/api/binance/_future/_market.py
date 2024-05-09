from typing import Dict

from ._data import BinanceFutureMarketDataEndPoints
from ._trade import BinanceFutureMarketTradeEndPoints
from ..._api.market import MarketEndPoints


class BinanceFutureMarketEndPoints(MarketEndPoints):
    def __init__(self, api_key: str, secret_key: str, proxies: Dict[str, str] = None):
        data = BinanceFutureMarketDataEndPoints(api_key=api_key, secret_key=secret_key, proxies=proxies)
        trade = BinanceFutureMarketTradeEndPoints(api_key=api_key, secret_key=secret_key, proxies=proxies)
        super().__init__(data=data, trade=trade)
