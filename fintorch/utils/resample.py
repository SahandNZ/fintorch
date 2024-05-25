from datetime import datetime

import pandas as pd

from .timestamp import floor_timestamp, ceil_timestamp
from ..enum import TimeFrame


def first_item(items):
    return list(items)[0]


def last_item(items):
    return list(items)[-1]


def resample_candles_df(
        df: pd.DataFrame,
        source_timeframe: TimeFrame,
        destination_timeframe: TimeFrame,
        inplace: bool = False
) -> pd.DataFrame:
    window_size = int(destination_timeframe) // int(source_timeframe)
    first_timestamp = ceil_timestamp(timestamp=df.index[0], time_frame=destination_timeframe)

    if not inplace:
        df = df.copy()

    df = df[first_timestamp <= df.index]
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    df["high"] = df.high.rolling(window=indexer, min_periods=1, step=window_size).max()
    df["low"] = df.low.rolling(window=indexer, min_periods=1, step=window_size).min()
    df["close"] = df.close.rolling(window=indexer, min_periods=1, step=window_size).agg(last_item)
    df["volume"] = df.volume.rolling(window=indexer, min_periods=1, step=window_size).sum()
    df['trade'] = df.trade.rolling(window=indexer, min_periods=1, step=window_size).sum()
    df = df.dropna()

    return df


def resample_long_short_ratio_df(
        df: pd.DataFrame,
        source_timeframe: TimeFrame,
        destination_timeframe: TimeFrame,
        inplace: bool = False
) -> pd.DataFrame:
    window_size = int(destination_timeframe) // int(source_timeframe)
    first_timestamp = ceil_timestamp(timestamp=df.index[0], time_frame=destination_timeframe)

    if not inplace:
        df = df.copy()

    df = df[first_timestamp <= df.index]
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    df['ratio'] = df.ratio.rolling(window=indexer, min_periods=1, step=window_size).agg(first_item)
    df['long'] = df.long.rolling(window=indexer, min_periods=1, step=window_size).agg(first_item)
    df['short'] = df.short.rolling(window=indexer, min_periods=1, step=window_size).agg(first_item)
    df = df.dropna()

    return df
