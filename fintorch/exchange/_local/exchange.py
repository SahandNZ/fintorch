from typing import List

from .data import LocalData
from .trade import LocalTrade
from .wallet import LocalWallet
from .. import Exchange, Market
from .._online import OnlineExchange
from ...enum import MarketType, TimeFrame


class LocalExchange(Exchange):
    def __init__(self, online_exchange: OnlineExchange, initial_capital: float, interval: TimeFrame):
        self.__online_exchange: Exchange = online_exchange

        name = f"Local {online_exchange.name} Exchange"
        wallet = LocalWallet(initial_capital=initial_capital)
        future_data = LocalData(market_type=MarketType.FUTURE, online_exchange=online_exchange, interval=interval)
        future_trade = LocalTrade(market_type=MarketType.FUTURE, data=future_data, wallet=wallet, interval=interval)
        future = Market(market_type=MarketType.FUTURE, data=future_data, trade=future_trade)

        super().__init__(name=name, wallet=wallet, spot=None, future=future)

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        super().prepare(symbols=symbols, time_frames=time_frames)
        self.__online_exchange.prepare(symbols=symbols, time_frames=time_frames)

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        self.__online_exchange.next(timestamp=timestamp)
