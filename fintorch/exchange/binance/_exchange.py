from typing import Dict

from ._future import BinanceFuture
from .._exchange import Exchange, Wallet, Future, Spot


class BinanceExchange(Exchange):
    def __init__(self, key: str = None, secret_key: str = None, proxies: Dict = None):
        wallet: Wallet = None
        spot: Spot = None
        future: Future = BinanceFuture(key=key, secret_key=secret_key, proxies=proxies)
        super().__init__(name="binance", wallet=wallet, spot=spot, future=future)
