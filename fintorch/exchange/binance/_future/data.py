from datetime import datetime
from typing import List, Any, Dict

from ._network.decorator import *
from ._network.deserializer import candle_deserializer, candle_wss_deserializer, symbol_info_deserializer
from ._network.encoder import *
from ._network.https import BinanceFutureHttps
from ._network.wss import BinanceFutureWss
from ... import OnlineData
from ....dtype import SymbolInfo, Candle
from ....enum import MarketType, TimeFrame


class BinanceFutureData(OnlineData):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame, key: str = None,
                 secret_key: str = None, proxies: Dict = None):
        https = BinanceFutureHttps(key=key, secret_key=secret_key, proxies=proxies)
        # wss = BinanceFutureWss(key=key, secret_key=secret_key, proxies=proxies)
        super().__init__(exchange_name=exchange_name, market_type=market_type, interval=interval, https=https, wss=None)

    @property
    def max_candles(self) -> int:
        return 1500

    def get_current_timestamp(self) -> int:
        endpoint = '/fapi/v1/time'
        response = self._https.get(endpoint=endpoint)
        server_time = int(response['serverTime'])
        return server_time

    def get_ping(self) -> int:
        local = datetime.now().timestamp() * 1000
        server = self.get_current_timestamp()
        ping = round(server - local)
        return ping

    @decode_symbol
    def _get_symbols_info(self) -> List[SymbolInfo]:
        endpoint = '/fapi/v1/exchangeInfo'
        response = self._https.get(endpoint=endpoint)
        perpetuals = [item for item in response['symbols'] if 'PERPETUAL' == item['contractType']]
        symbols_info = [symbol_info_deserializer(item) for item in perpetuals]
        return symbols_info

    @encode_symbol
    @encode_time_frame
    def _get_recent_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        endpoint = '/fapi/v1/klines'
        params = {'symbol': symbol, 'interval': time_frame}
        response = self._https.get(endpoint=endpoint, params=params)
        candles = [candle_deserializer(item) for item in response]
        return candles

    @encode_symbol
    def _get_historical_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp) \
            -> List[Candle]:
        endpoint = '/fapi/v1/klines'
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
