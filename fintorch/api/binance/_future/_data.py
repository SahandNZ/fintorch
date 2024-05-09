from typing import List

from .network.decorator import *
from .network.deserializer import *
from .network.https import BinanceFutureHttps
from .network.wss import BinanceFutureWss
from ..._api.market import MarketDataEndPoints
from ....dtype import SymbolInfo, Candle, FundingRate, LongShortRatio, AggregatedTrade
from ....enum import TimeFrame


class BinanceFutureMarketDataEndPoints(MarketDataEndPoints):
    def __init__(self, api_key: str = None, secret_key: str = None, proxies: Dict = None):
        https = BinanceFutureHttps(api_key=api_key, secret_key=secret_key, proxies=proxies)
        wss = BinanceFutureWss(api_key=api_key, secret_key=secret_key, proxies=proxies)
        super().__init__(https=https, wss=wss)

    @property
    def _max_aggregated_trades_request_size(self) -> int:
        return 1000

    @property
    def _max_candles_request_size(self) -> int:
        return 1500

    @property
    def _max_funding_rates_request_size(self) -> int:
        return 1000

    @property
    def funding_rates_interval(self) -> TimeFrame:
        return TimeFrame.HOUR1 * 8

    @property
    def _max_top_long_short_ratios_account_request_size(self) -> int:
        return 500

    def _get_server_timestamp(self) -> float:
        endpoint = '/fapi/v1/time'
        response = self._https.get(endpoint=endpoint)
        server_time = int(response['serverTime']) / 1000
        return server_time

    @decode_symbol
    def _get_symbols_info(self) -> List[SymbolInfo]:
        endpoint = '/fapi/v1/exchangeInfo'
        response = self._https.get(endpoint=endpoint)
        perpetuals = [item for item in response['symbols'] if 'PERPETUAL' == item['contractType']]
        symbols_info = [symbol_info_deserializer(item) for item in perpetuals]
        return symbols_info

    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        symbols_info = self._get_symbols_info()
        symbol_info = [symbol_info for symbol_info in symbols_info if symbol_info.symbol == symbol][0]
        return symbol_info

    @decode_symbol
    def _get_symbols_ticker(self) -> List[Ticker]:
        endpoint = '/fapi/v2/ticker/price'
        response = self._https.get(endpoint=endpoint)
        tickers = [ticker_deserializer(item) for item in response]
        return tickers

    @decode_symbol
    def _get_symbol_ticker(self, symbol: str) -> Ticker:
        endpoint = '/fapi/v2/ticker/price'
        params = {"symbol": symbol}
        response = self._https.get(endpoint=endpoint, params=params)
        tickers = [ticker_deserializer(item) for item in response if symbol_decoder(item["symbol"]) == symbol][0]
        return tickers

    @encode_symbol
    @decode_symbol
    def _get_aggregated_trades(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: float,
            stop_timestamp: float
    ) -> List[AggregatedTrade]:
        endpoint = '/fapi/v1/aggTrades'
        params = {
            "symbol": symbol,
            "startTime": int(start_timestamp * 1000),
            "stopTime": int(stop_timestamp * 1000),
            "limit": self._max_aggregated_trades_request_size
        }
        response = self._https.get(endpoint=endpoint, params=params)
        tickers = [aggregated_trade_deserializer(item) for item in response]
        return tickers

    @encode_symbol
    def _get_candles(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: float,
            stop_timestamp: float
    ) -> List[Candle]:
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

    @encode_symbol
    def _get_funding_rates(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: float,
            stop_timestamp: float
    ) -> List[FundingRate]:
        endpoint = '/fapi/v1/fundingRate'
        limit = (stop_timestamp - start_timestamp) // self.funding_rates_interval
        params = {
            'symbol': symbol,
            'startTime': int(start_timestamp * 1000),
            'endTime': int(stop_timestamp * 1000),
            'limit': int(limit)
        }
        response = self._https.get(endpoint=endpoint, params=params)
        funding_rates = [funding_rate_deserializer(item) for item in response]
        return funding_rates

    @encode_symbol
    def _get_top_long_short_ratios_account(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: float,
            stop_timestamp: float
    ) -> List[LongShortRatio]:
        endpoint = '/futures/data/globalLongShortAccountRatio'
        limit = (stop_timestamp - start_timestamp) // time_frame
        params = {
            'symbol': symbol,
            'period': time_frame_encoder(time_frame),
            'startTime': int(start_timestamp * 1000),
            'endTime': int(stop_timestamp * 1000),
            'limit': int(limit)
        }
        response = self._https.get(endpoint=endpoint, params=params)
        long_short_ratios = [long_short_ratio_deserializer(item) for item in response]
        return long_short_ratios
