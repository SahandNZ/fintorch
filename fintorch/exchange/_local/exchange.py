from .market import LocalMarket
from .. import Exchange
from .._online import OnlineExchange
from ...enum import MarketType, TimeFrame


class LocalExchange(Exchange):
    def __init__(self, online_exchange: OnlineExchange, interval: TimeFrame):
        self.__online_exchange: OnlineExchange = online_exchange

        name = f"Local {online_exchange.name.split(' ')[1]} Exchange"
        account = None
        spot = None
        future = LocalMarket(online_exchange=online_exchange, interval=interval, market_type=MarketType.FUTURE)
        super().__init__(name=name, account=account, spot=spot, future=future)

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        self.__online_exchange.next(timestamp=timestamp)
