import os
from abc import ABC
from typing import List

from .account import Account
from .market import Market
from ...enum import TimeFrame


class Exchange(ABC):
    def __init__(self, name: str, account: Account, spot: Market, future: Market):
        self.__name: str = name
        self.__account: Account = account
        self.__spot: Market = spot
        self.__future: Market = future

    @property
    def name(self) -> str:
        return self.__name

    @property
    def account(self) -> Account:
        return self.__account

    @property
    def spot(self) -> Market:
        return self.__spot

    @property
    def future(self) -> Market:
        return self.__future

    def open(self, directory: str) -> None:
        directory = os.path.join(directory, self.name.replace(" ", "-").lower())
        # TODO call open methods of account and spot
        self.future.open(directory=directory)

    def next(self, timestamp: float) -> None:
        # TODO call next methods of account and spot
        self.future.next(timestamp=timestamp)

    def close(self, directory: str) -> None:
        directory = os.path.join(directory, self.name.replace(" ", "-").lower())
        # TODO call close methods of account and spot
        self.future.close(directory=directory)
