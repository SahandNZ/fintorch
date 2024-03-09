from typing import Dict

from .wallet import Wallet
from .. import Exchange, Market
from ...enum import TimeFrame
from ...utils.function import call_with_dict, import_class


class OnlineExchange(Exchange):
    def __init__(self, name: str, wallet: Wallet, spot: Market, future: Market):
        super().__init__(name=name, wallet=wallet, spot=spot, future=future)

    @staticmethod
    def from_name(name: str, interval: TimeFrame, proxies: Dict[str, str] = None):
        dct = {"exchange-name": name, "interval": interval, "proxies": proxies}
        exchange = OnlineExchange.from_dict(dct=dct)

        return exchange

    @staticmethod
    def from_dict(dct: Dict):
        name = dct["exchange-name"]
        exchange_cls = import_class(module=f"fintorch.exchange.{name}", contains=f"{name}exchange")
        return call_with_dict(exchange_cls, dct)
