from typing import Dict

from ._future import BinanceFutureData, BinanceFutureTrade
from .. import Market, Exchange
from ...enum import MarketType, TimeFrame


class BinanceExchange(Exchange):
    def __init__(self, interval: TimeFrame, key: str = None, secret_key: str = None, proxies: Dict = None):
        future_data = BinanceFutureData(
            exchange_name="binance",
            market_type=MarketType.FUTURE,
            interval=interval,
            key=key,
            secret_key=secret_key,
            proxies=proxies
        )
        future_trade = BinanceFutureTrade(
            exchange_name="binance",
            market_type=MarketType.FUTURE,
            interval=interval,
            key=key,
            secret_key=secret_key,
            proxies=proxies
        )
        future_market = Market(market_type=MarketType.FUTURE, data=future_data, trade=future_trade)
        super().__init__(name="binance", wallet=None, spot=None, future=future_market)
