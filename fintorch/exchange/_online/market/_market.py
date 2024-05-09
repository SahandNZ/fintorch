from ._data import OnlineMarketData
from ._trade import OnlineMarketTrade
from ..._exchange.market import Market
from ....api import API
from ....enum import MarketType


class OnlineMarket(Market):
    def __init__(self, api: API, market_type: MarketType):
        data = OnlineMarketData(api=api, market_type=market_type)
        trade = OnlineMarketTrade(api=api, market_type=market_type)
        super().__init__(data=data, trade=trade, market_type=market_type)
