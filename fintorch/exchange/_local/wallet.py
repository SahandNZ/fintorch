from typing import Dict

from ...dtype import Balance
from ...enum import MarketType
from ...exchange import Wallet


class LocalWallet(Wallet):
    def __init__(self, initial_capital: float):
        super().__init__()
        balance = Balance()
        balance.asset = "USDT"
        balance.total = initial_capital
        balance.frozen = 0

        market_type_to_asset_to_balance = {MarketType.FUTURE: {balance.asset: balance}}
        self.__market_to_asset_to_balance: Dict[MarketType, Dict[str, Balance]] = market_type_to_asset_to_balance

    def get_balance(self, market_type: MarketType, asset: str) -> Balance:
        if market_type not in self.__market_to_asset_to_balance:
            self.__market_to_asset_to_balance[market_type] = {}

        if asset not in self.__market_to_asset_to_balance[market_type]:
            balance = Balance()
            balance.asset = asset
            balance.total = 0
            balance.frozen = 0

            self.__market_to_asset_to_balance[market_type][balance.asset] = balance

        return self.__market_to_asset_to_balance[market_type][asset]

    def transfer_balance(self, source: MarketType, destination: MarketType, asset: str, amount: float) -> bool:
        source_balance = self.get_balance(market_type=source, asset=asset)
        destination_balance = self.get_balance(market_type=destination, asset=asset)
        if source_balance.available < amount:
            raise Exception("Insufficient balance")
        else:
            source_balance.total -= amount
            destination_balance.total += amount
            return True
