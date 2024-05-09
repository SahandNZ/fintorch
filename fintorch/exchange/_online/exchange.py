from .market import OnlineMarket
from .. import Exchange
from ...api import API
from ...enum import MarketType


class OnlineExchange(Exchange):
    def __init__(self, api: API):
        name = f"Online {api.name.title()} Exchange"
        account = None
        spot = None
        future = OnlineMarket(api=api, market_type=MarketType.FUTURE)
        super().__init__(name=name, account=account, spot=spot, future=future)
