from abc import ABC

from .account import AccountEndPoints
from .market import MarketEndPoints


class API(ABC):
    def __init__(self, name: str, account: AccountEndPoints, spot: MarketEndPoints, future: MarketEndPoints) -> None:
        self.__name: str = name
        self.__account: AccountEndPoints = account
        self.__spot: MarketEndPoints = spot
        self.__future: MarketEndPoints = future

    @property
    def name(self) -> str:
        return self.__name

    @property
    def account(self) -> AccountEndPoints:
        return self.__account

    @property
    def spot(self) -> MarketEndPoints:
        return self.__spot

    @property
    def future(self) -> MarketEndPoints:
        return self.__future
