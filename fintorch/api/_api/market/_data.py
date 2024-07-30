import math
from abc import abstractmethod
from datetime import datetime
from typing import List, Union, Type

from rich.progress import Progress

from ..network import Network, Https, Wss
from ....dtype import AggregatedTrade, SymbolInfo, Candle, FundingRate, LongShortRatio, Ticker
from ....enum import TimeFrame
from ....utils.timestamp import ceil_timestamp


class MarketDataEndPoints(Network):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https, wss)

    def get_server_timestamp(self) -> float:
        return self._get_server_timestamp()

    def get_ping(self) -> float:
        local = datetime.now().timestamp()
        server = self.get_server_timestamp()
        ping = round(server - local)
        return ping

    def get_symbols_info(self) -> List[SymbolInfo]:
        return self._get_symbols_info()

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self._get_symbol_info(symbol=symbol)

    def get_symbols_ticker(self) -> List[Ticker]:
        return self._get_symbols_ticker()

    def get_symbol_ticker(self, symbol: str) -> Ticker:
        return self._get_symbol_ticker(symbol=symbol)

    def get_aggregated_trades(
            self,
            symbol: str,
            start_timestamp: int,
            stop_timestamp: int,
            progress: Union[Progress, None] = None,
    ):
        return self.__send_get_requests(
            symbol=symbol,
            time_frame=self.aggregated_trades_interval,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            attribute_name="aggregated_trades",
            dtype=Candle,
            progress=progress
        )

    def get_candles(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: Union[int, None],
            progress: Union[Progress, None] = None,
    ) -> List[Candle]:
        return self.__send_get_requests(
            symbol=symbol,
            time_frame=time_frame,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            attribute_name="candles",
            dtype=Candle,
            progress=progress
        )

    def get_funding_rates(
            self,
            symbol: str,
            start_timestamp: int,
            stop_timestamp: int,
            progress: Union[Progress, None] = None,
    ) -> List[FundingRate]:
        return self.__send_get_requests(
            symbol=symbol,
            time_frame=self.funding_rates_interval,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            attribute_name="funding_rates",
            dtype=FundingRate,
            progress=progress
        )

    def get_top_long_short_ratios_account(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: int,
            progress: Union[Progress, None] = None
    ) -> List[LongShortRatio]:
        return self.__send_get_requests(
            symbol=symbol,
            time_frame=time_frame,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            attribute_name="top_long_short_ratios_account",
            dtype=LongShortRatio,
            progress=progress
        )

    # region child abstractmethod properties and methods
    @property
    @abstractmethod
    def _max_aggregated_trades_request_size(self) -> int:
        raise NotImplementedError()

    @property
    def aggregated_trades_interval(self) -> TimeFrame:
        return TimeFrame.MS100

    @property
    @abstractmethod
    def _max_candles_request_size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def _max_funding_rates_request_size(self) -> int:
        raise NotImplementedError()

    @property
    def funding_rates_interval(self) -> TimeFrame:
        return TimeFrame.HOUR1 * 8

    @property
    @abstractmethod
    def _max_top_long_short_ratios_account_request_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _get_server_timestamp(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _get_symbols_info(self) -> List[SymbolInfo]:
        raise NotImplementedError()

    @abstractmethod
    def _get_symbol_info(self, symbol: str) -> SymbolInfo:
        raise NotImplementedError()

    @abstractmethod
    def _get_symbols_ticker(self) -> List[Ticker]:
        raise NotImplementedError()

    @abstractmethod
    def _get_symbol_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError()

    @abstractmethod
    def _get_aggregated_trades(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: int
    ) -> List[AggregatedTrade]:
        raise NotImplementedError()

    @abstractmethod
    def _get_candles(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: int
    ) -> List[Candle]:
        raise NotImplementedError()

    @abstractmethod
    def _get_funding_rates(self, symbol: str, start_timestamp: int, stop_timestamp: int) -> List[FundingRate]:
        raise NotImplementedError()

    @abstractmethod
    def _get_top_long_short_ratios_account(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: int
    ) -> List[LongShortRatio]:
        raise NotImplementedError()

    # endregion

    # region private methods
    def __send_get_requests(
            self,
            symbol: str,
            time_frame: TimeFrame,
            start_timestamp: int,
            stop_timestamp: int,
            attribute_name: str,
            dtype: Type[Union[FundingRate, Candle, LongShortRatio]],
            progress: Union[Progress, None] = None,
    ) -> List:
        max_request_size = getattr(self, f"_max_{attribute_name}_request_size")
        get_request_fn = getattr(self, f"_get_{attribute_name}")

        # Setup progress bar if it's not none
        if progress is not None:
            current_timestamp = int(datetime.now().timestamp())
            next_open_timestamp = ceil_timestamp(timestamp=current_timestamp, time_frame=time_frame)
            last_timestamp = stop_timestamp or next_open_timestamp
            total = math.ceil((last_timestamp - start_timestamp) / int(time_frame) / max_request_size)
            description = f"Downloading {symbol} {str(time_frame)} {attribute_name.replace('_', ' ')}"
            task = progress.add_task(description=description, total=total)
        else:
            task = None

        results: List[dtype] = []
        request_start_timestamp = start_timestamp
        request_stop_timestamp = request_start_timestamp + max_request_size * time_frame
        while request_start_timestamp < request_stop_timestamp:
            request_results = get_request_fn(
                symbol=symbol,
                time_frame=time_frame,
                start_timestamp=request_start_timestamp,
                stop_timestamp=request_stop_timestamp
            )
            results.extend(request_results)

            # update progress bar
            if progress is not None:
                progress.update(task, advance=1)

            # assign value to request start timestamp
            if 0 == len(results):
                request_start_timestamp += max_request_size * time_frame
            else:
                request_start_timestamp = results[-1].timestamp + time_frame

            # assign value to request stop timestamp
            current_timestamp = int(datetime.now().timestamp())
            next_open_timestamp = ceil_timestamp(timestamp=current_timestamp, time_frame=time_frame)
            last_timestamp = stop_timestamp or next_open_timestamp
            request_stop_timestamp = request_start_timestamp + max_request_size * time_frame
            request_stop_timestamp = min(last_timestamp, request_stop_timestamp)

        # make progress bar invisible
        if progress is not None:
            progress.update(task, visible=False)

        return results
    # endregion
