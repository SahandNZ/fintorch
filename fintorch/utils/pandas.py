import pandas as pd
from .timestamp import round_timestamp

def resample_df(
    df: pd.DataFrame,
    source_timeframe: int,
    destination_timeframe: int,
    inplace: bool=False
) -> pd.DataFrame:
        
    window_size = destination_timeframe // source_timeframe
    first_timestamp = round_timestamp(timestamp=df.index[0], time_frame=destination_timeframe)
    first_timestamp += destination_timeframe 
    
    if inplace:
        df = df.copy()
    
    df = df[first_timestamp <= df.index]
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    df["high"] = df.high.rolling(window=indexer, min_periods=1, step=window_size).max()
    df["low"] = df.low.rolling(window=indexer, min_periods=1, step=window_size).min()
    df["close"] = df.close.rolling(window=indexer, min_periods=1, step=window_size).agg(lambda items: list(items)[-1])
    df["volume"] = df.volume.rolling(window=indexer, min_periods=1, step=window_size).sum()
    if "trade" in df.columns:
        df['trade'] = df.trade.rolling(window=indexer, min_periods=1, step=window_size).sum()
    df = df.dropna()

    return df
