from typing import List, Dict

from fintorch.dtype import Ticker

from ._network.encoder import *
from ._network.decoder import *
from ._network.decorator import *
from ._network.deserializer import *
from ._network.https import BinanceFutureHttps
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
        server_time = int(response['serverTime']) // 1000
        return server_time

    @decode_symbol
    def _get_symbols_info(self) -> List[SymbolInfo]:
        endpoint = '/fapi/v1/exchangeInfo'
        response = self._https.get(endpoint=endpoint)
        perpetuals = [item for item in response['symbols'] if 'PERPETUAL' == item['contractType']]
        symbols_info = [symbol_info_deserializer(item) for item in perpetuals]
        return symbols_info
    
    @decode_symbol
    def get_symbols_ticker(self, symbols: List[str]) -> Ticker:
        endpoint = '/fapi/v2/ticker/price'
        response = self._https.get(endpoint=endpoint)
        tickers = [tikcer_deserializer(item) for item in response if symbol_decoder(item["symbol"]) in symbols]
        return tickers

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
