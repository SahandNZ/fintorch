import itertools
from datetime import datetime
from typing import List, Tuple

from ..enum import MarketType, TimeFrame
from ..exchange import OnlineData, OnlineExchange


class Context:
    def __init__(
            self,
            online_exchange: OnlineExchange,
            market_type: MarketType,
            symbols: List[str],
            time_frames: List[TimeFrame]
    ):
        self.__online_exchange: OnlineExchange = online_exchange
        self.__market_type: MarketType = market_type
        self.__symbols: List[str] = symbols
        self.__time_frames: List[TimeFrame] = time_frames
        self.__pairs: List[Tuple[str, TimeFrame]] = list(itertools.product(self.symbols, self.time_frames))

        self.__online_data: OnlineData = getattr(self.online_exchange, str(market_type)).data
        self.online_exchange.prepare(symbols=self.symbols, time_frames=self.time_frames)

    @property
    def online_exchange(self) -> OnlineExchange:
        return self.__online_exchange

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def online_data(self) -> OnlineData:
        return self.__online_data

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.__time_frames

    @property
    def pairs(self) -> List[Tuple[str, TimeFrame]]:
        return self.__pairs

    def refresh(self):
        minimum_time_frame = min([int(tf) for tf in self.time_frames])
        current_open_timestamp = int(datetime.now().timestamp() // minimum_time_frame * minimum_time_frame)
        self.online_data.next(timestamp=current_open_timestamp)
