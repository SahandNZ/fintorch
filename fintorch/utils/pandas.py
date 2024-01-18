import pandas as pd


def resample_time_frame(tohlcv: pd.DataFrame, source_timeframe: int, destination_timeframe: int) -> pd.DataFrame:
    df = tohlcv.copy()
    step = destination_timeframe // source_timeframe

    df['is_first'] = 0 == (df.index.to_series() % destination_timeframe)
    df['is_last'] = df.is_first.shift(step - 1).fillna(False)

    df['open'] = df.open[df.is_first]
    df['high'] = df.high.rolling(step).max().shift(-step + 1)
    df['low'] = df.low.rolling(step).min().shift(-step + 1)
    df['close'] = df.close[df.is_last]
    df['close'] = df.close.bfill()
    df['volume'] = df.volume.rolling(step).sum().shift(-step + 1)

    df = df.dropna()
    df = df.drop(['is_first', 'is_last'], axis=1)

    return df
