from datetime import datetime
from typing import List, Any, Dict

from .decorator import *
from .deserializer import candle_deserializer, candle_wss_deserializer, symbol_info_deserializer
from .encoder import *
from ..._exchange import Https, Wss, Market
from ....dtype import SymbolInfo, Candle
from ....enum import TimeFrame


class BinanceFutureMarket(Market):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https, wss)

    @property
    def max_candles(self) -> int:
        return 1500

    def get_server_time(self) -> int:
        endpoint = '/fapi/v1/time'
        response = self._https.get(endpoint=endpoint)
        server_time = int(response['serverTime'])
        return server_time

    def get_ping(self) -> int:
        local = datetime.now().timestamp() * 1000
        server = self.get_server_time()
        ping = round(server - local)
        return ping

    @decode_symbol
    def get_symbols_info(self) -> List[SymbolInfo]:
        endpoint = '/fapi/v1/exchangeInfo'
        response = self._https.get(endpoint=endpoint)
        perpetuals = [item for item in response['symbols'] if 'PERPETUAL' == item['contractType']]
        symbols_info = [symbol_info_deserializer(item) for item in perpetuals]
        return symbols_info

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        symbols_info = self.get_symbols_info()
        symbol_info = [item for item in symbols_info if item.symbol == symbol][0]
        return symbol_info

    @encode_symbol
    @encode_time_frame
    def get_recent_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        endpoint = '/fapi/v1/klines'
        params = {'symbol': symbol, 'interval': time_frame}
        response = self._https.get(endpoint=endpoint, params=params)
        candles = [candle_deserializer(item) for item in response]
        return candles

    @encode_symbol
    def get_historical_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp) \
            -> List[Candle]:
        endpoint = '/fapi/v1/klines'
        limit = (stop_timestamp - start_timestamp) // time_frame
        params = {'symbol': symbol,
                  'interval': time_frame_encoder(time_frame),
                  'startTime': int(start_timestamp * 1000),
                  'endTime': int(stop_timestamp * 1000),
                  'limit': int(limit)}
        response = self._https.get(endpoint=endpoint, params=params)
        candles = [candle_deserializer(item) for item in response]

        return candles

    @encode_symbol
    @encode_time_frame
    def subscribe_candles(self, symbol: str, time_frame: TimeFrame, on_message: Callable[[Candle], Any]):
        def pre_on_message(message: Dict):
            data = message["k"]
            candle = candle_wss_deserializer(data)
            on_message(candle)

        self._wss.subscribe_stream(stream=f"{symbol.lower()}@kline_{time_frame}", on_message=pre_on_message)
