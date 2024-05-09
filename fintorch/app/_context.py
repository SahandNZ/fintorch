import itertools
from datetime import datetime
from typing import List, Tuple

from ..enum import MarketType, TimeFrame
from ..exchange import OnlineMarketData, OnlineExchange


class Context:
    def __init__(
            self,
            symbols: List[str],
            time_frames: List[TimeFrame],
            online_exchange: OnlineExchange,
            market_type: MarketType,
    ):
        self.__symbols: List[str] = symbols
        self.__time_frames: List[TimeFrame] = time_frames
        self.__online_exchange: OnlineExchange = online_exchange
        self.__market_type: MarketType = market_type

        self.__pairs: List[Tuple[str, TimeFrame]] = list(itertools.product(self.symbols, self.time_frames))
        self.__online_market_data: OnlineMarketData = getattr(self.online_exchange, str(market_type)).data

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.__time_frames

    @property
    def online_exchange(self) -> OnlineExchange:
        return self.__online_exchange

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def pairs(self) -> List[Tuple[str, TimeFrame]]:
        return self.__pairs

    @property
    def online_market_data(self) -> OnlineMarketData:
        return self.__online_market_data

    def refresh(self):
        minimum_time_frame = min([int(tf) for tf in self.time_frames])
        current_open_timestamp = int(datetime.now().timestamp() // minimum_time_frame * minimum_time_frame)
        self.online_market_data.next(timestamp=current_open_timestamp)
