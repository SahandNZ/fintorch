from typing import Dict

from .future import Future
from .spot import Spot
from .wallet import Wallet
from ...utils.function import call_with_dict, import_class


class Exchange:
    def __init__(self, name: str, wallet: Wallet, spot: Spot, future: Future):
        self.__name: str = name
        self.__wallet: Wallet = wallet
        self.__spot: Spot = spot
        self.__future: Future = future

    @staticmethod
    def from_name(name: str, proxies: Dict[str, str] = None):
        dct = {"name": name, "proxies": proxies}
        exchange = Exchange.from_dict(dct=dct)

        return exchange

    @staticmethod
    def from_dict(dct: Dict):
        name = dct["name"]
        exchange_cls = import_class(module=f"fintorch.exchange.{name}", contains=f"{name}exchange")
        return call_with_dict(exchange_cls, dct)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def wallet(self) -> Wallet:
        return self.__wallet

    @property
    def spot(self) -> Spot:
        return self.__spot

    @property
    def future(self) -> Future:
        return self.__future
