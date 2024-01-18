import itertools
import os
from typing import Dict, List, Tuple

import pandas as pd
from rich.progress import Progress

from ._live import LiveData
from ._local import LocalData
from ..dtype import Candle, Data
from ..enum import TimeFrame
from ..exchange import EXCHANGE
from ..setting import EXCHANGE_NAME, BASE_TIME_FRAME, CANDLE_DIR
from ..utils.pandas import resample_time_frame

LOCAL_DATA = LocalData(exchange=EXCHANGE)


def get_local_symbols() -> List[str]:
    root_dir = os.path.join(CANDLE_DIR, EXCHANGE_NAME)
    symbols = os.listdir(root_dir)
    return symbols


def get_exchange_symbols() -> List[str]:
    symbols_info = EXCHANGE.future.market.get_symbols_info()
    symbols = [symbol_info.symbol for symbol_info in symbols_info]
    return symbols


def load_dataframe(symbol: str, time_frame: TimeFrame, update: bool = False) -> pd.DataFrame:
    if update:
        candles = LOCAL_DATA.download_candles(symbol=symbol, time_frame=BASE_TIME_FRAME)
        df = Candle.to_dataframe(candles=candles)
    else:
        df = LOCAL_DATA.load_dataframe(symbol=symbol, time_frame=BASE_TIME_FRAME)

    if time_frame < BASE_TIME_FRAME:
        raise ValueError("time frame must be greater than {}".format(time_frame))

    if BASE_TIME_FRAME < time_frame:
        df = resample_time_frame(tohlcv=df, source_timeframe=BASE_TIME_FRAME, destination_timeframe=time_frame)

    return df


def load_dataframes_dict(symbols: List[str], time_frames: List[TimeFrame], update: bool = False,
                         progress: Progress = None) -> Dict[Tuple[str, int], pd.DataFrame]:
    items = list(itertools.product(symbols, time_frames))

    if progress is not None:
        task = progress.add_task(description="Loading DataFrames", total=len(items))

    dfs_dict = {}
    for symbol, time_frame in items:
        df = load_dataframe(symbol=symbol, time_frame=time_frame, update=update)
        dfs_dict[symbol, time_frame] = df

        if progress is not None:
            progress.update(task, advance=1)

    return dfs_dict


def load_data(symbols: List, time_frames: List, update: bool = False, progress: Progress = None) -> Data:
    df_dict = load_dataframes_dict(symbols=symbols, time_frames=time_frames, update=update, progress=progress)
    data = Data(df_dict)

    return data
