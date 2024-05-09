from typing import Dict

from ._future import BinanceFutureMarketEndPoints
from ...api import API


class BinanceAPI(API):
    def __init__(self, api_key: str = None, secret_key: str = None, proxies: Dict = None):
        account = None
        spot = None
        future = BinanceFutureMarketEndPoints(api_key=api_key, secret_key=secret_key, proxies=proxies)
        super().__init__(name="binance", account=account, spot=spot, future=future)
