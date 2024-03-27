from typing import List, Dict

from ._network.decorator import *
from ._network.deserializer import candle_deserializer, symbol_info_deserializer
from ._network.encoder import *
from ._network.https import BingxFutureHttps
from ... import OnlineData
from ....dtype import SymbolInfo, Candle
from ....enum import MarketType, TimeFrame


class BingxFutureData(OnlineData):
    def __init__(
            self,
            exchange_name: str,
            market_type: MarketType,
            interval: TimeFrame,
            key: str = None,
            secret_key: str = None,
            proxies: Dict = None
    ):
        https = BingxFutureHttps(key=key, secret_key=secret_key, proxies=proxies)
        super().__init__(exchange_name=exchange_name, market_type=market_type, interval=interval, https=https, wss=None)

    @property
    def max_candles(self) -> int:
        return 1440

    def get_current_timestamp(self) -> int:
        endpoint = "/openApi/swap/v2/server/time"
        response = self._https.get(endpoint=endpoint)
        server_time = int(response['serverTime']) // 1000
        return server_time

    def _get_symbols_info(self) -> List[SymbolInfo]:
        endpoint = "/openApi/swap/v2/quote/contracts"
        response = self._https.get(endpoint=endpoint)
        symbols_info = [symbol_info_deserializer(item) for item in response]
        return symbols_info

    @encode_time_frame
    def _get_recent_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        raise NotImplementedError()

    def _get_historical_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp) \
            -> List[Candle]:
        endpoint = "/openApi/swap/v1/market/markPriceKlines"
        limit = (stop_timestamp - start_timestamp) // time_frame
        params = {
            'symbol': symbol,
            'interval': time_frame_encoder(time_frame),
            'startTime': int(start_timestamp * 1000),
            'endTime': int(stop_timestamp * 1000),
            'limit': int(limit)
        }
        response = self._https.get(endpoint=endpoint, params=params)
        candles = [candle_deserializer(item) for item in response]

        return candles
