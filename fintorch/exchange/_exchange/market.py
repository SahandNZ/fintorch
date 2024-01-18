import math
from abc import abstractmethod
from datetime import datetime
from typing import List, Any, Callable, Union

from rich.progress import Progress

from .https import Https
from .network import Network
from .wss import Wss
from ...setting import RICH_PROGRESS_COLUMNS
from ...dtype import Candle, SymbolInfo
from ...enum import TimeFrame


class Market(Network):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https, wss)

    @property
    @abstractmethod
    def max_candles(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_server_time(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_ping(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_symbols_info(self) -> List[SymbolInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_recent_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        raise NotImplementedError()

    @abstractmethod
    def get_historical_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp) -> \
            List[Candle]:
        raise NotImplementedError()

    def get_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp: int = None,
                    progress: Union[Progress, bool] = None):
        if stop_timestamp is None:
            stop_timestamp = datetime.now().timestamp() // time_frame * time_frame + time_frame
        candles_count = (stop_timestamp - start_timestamp) // time_frame
        requests_count = math.ceil(candles_count / self.max_candles)

        items = list(range(requests_count))
        if isinstance(progress, bool) and progress:
            progress = Progress(*RICH_PROGRESS_COLUMNS)
        if isinstance(progress, Progress):
            description = f"Downloading {symbol} {time_frame} Candles"
            task = progress.add_task(description=description, total=len(items))

        candles: List[Candle] = []
        for index in items:
            # assign value to req_start_ts as request start timestamp
            if 0 == len(candles):
                req_start_ts = start_timestamp + index * self.max_candles * time_frame
            else:
                req_start_ts = candles[-1].timestamp + time_frame

            # assign value to req_stop_ts as request stop timestamp
            req_stop_ts = req_start_ts + self.max_candles * time_frame
            if stop_timestamp < req_stop_ts:
                req_stop_ts = stop_timestamp

            # send request
            if req_start_ts < req_stop_ts:
                req_candles = self.get_historical_candles(symbol=symbol, time_frame=time_frame,
                                                          start_timestamp=req_start_ts, stop_timestamp=req_stop_ts)
                candles.extend(req_candles)

            # update progress bar
            if progress is not None:
                with progress:
                    progress.update(task, advance=1)

        return candles

    @abstractmethod
    def subscribe_candles(self, symbol: str, time_frame: TimeFrame, on_message: Callable[[Candle], Any]):
        raise NotImplementedError()

    def join_wss(self):
        self._wss.join()
