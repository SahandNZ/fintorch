from typing import Dict

from ._future import BingxFutureData, BingxFutureTrade
from .. import Market, Exchange
from ...enum import MarketType, TimeFrame


class BingxExchange(Exchange):
    def __init__(self, interval: TimeFrame, key: str = None, secret_key: str = None, proxies: Dict = None):
        exchange_name = "bingx"
        future_data = BingxFutureData(
            exchange_name=exchange_name,
            market_type=MarketType.FUTURE,
            interval=interval,
            key=key,
            secret_key=secret_key,
            proxies=proxies
        )
        future_trade = BingxFutureTrade(
            exchange_name=exchange_name,
            market_type=MarketType.FUTURE,
            interval=interval,
            key=key,
            secret_key=secret_key,
            proxies=proxies
        )
        future_market = Market(market_type=MarketType.FUTURE, data=future_data, trade=future_trade)
        super().__init__(name=exchange_name, wallet=None, spot=None, future=future_market)
