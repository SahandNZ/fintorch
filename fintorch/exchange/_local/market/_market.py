from ._data import LocalMarketData
from ._trade import LocalMarketTrade
from ..._exchange.market import Market
from ..._online import OnlineExchange
from ....enum import MarketType, TimeFrame


class LocalMarket(Market):
    def __init__(
            self,
            online_exchange: OnlineExchange,
            interval: TimeFrame,
            market_type: MarketType
    ) -> None:
        data = LocalMarketData(online_exchange=online_exchange, market_type=market_type)
        trade = LocalMarketTrade(interval=interval, data=data, market_type=market_type)
        super().__init__(data=data, trade=trade, market_type=market_type)
