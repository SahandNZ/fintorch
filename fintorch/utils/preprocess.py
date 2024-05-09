import numpy as np
import pandas as pd


def add_heikin_ashi(df: pd.DataFrame, price_precision: int = None, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    ha_close_values = df[["open", "high", "low", "close"]].mean(axis=1).to_list()
    ha_open_values = [(df.open.iloc[0] + df.close.iloc[0]) / 2]
    for index in range(1, len(df)):
        ha_open_values.append((ha_open_values[index - 1] + ha_close_values[index - 1]) / 2)

    df["ha-close"] = ha_close_values
    df["ha-open"] = ha_open_values
    df["ha-high"] = df[["high", "ha-open", "ha-close"]].max(axis=1)
    df["ha-low"] = df[["low", "ha-open", "ha-close"]].min(axis=1)
    df = df.dropna()

    if price_precision is not None:
        df["ha-open"] = np.round(df["ha-open"], price_precision)
        df["ha-high"] = np.round(df["ha-high"], price_precision)
        df["ha-low"] = np.round(df["ha-low"], price_precision)
        df["ha-close"] = np.round(df["ha-close"], price_precision)

    return df


def add_fractals(df: pd.DataFrame, length: int, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    df["center-max"] = df.close.rolling(window=length, min_periods=1, center=True).max()
    df["center-min"] = df.close.rolling(window=length, min_periods=1, center=True).min()
    df["is-high-fractal"] = df.close == df["center-max"]
    df["is-low-fractal"] = df.close == df["center-min"]
    df["is-fractal"] = df["is-high-fractal"] | df["is-low-fractal"]
    df["fractal"] = df.close[df["is-fractal"]]

    return df


def remove_price_dependency(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    df["low-open"] = df.low / df.open - 1
    df["high-open"] = df.high / df.open - 1
    df["close-open"] = df.close / df.open - 1

    df["close"] = np.cumsum(df["close-open"])
    df["open"] = df.close.shift(1)
    df["low"] = df.open + df["low-open"]
    df["high"] = df.open + df["high-open"]

    df = df.drop(columns=["low-open", "high-open", "close-open"])
    df = df.dropna()

    return df


def fft(array: np.array, muting_percentage: int) -> np.array:
    f = array
    n = len(f)
    f_hat = np.fft.fft(f, n)

    psd = np.real(f_hat * np.conj(f_hat) / n)
    psd_threshold = np.percentile(psd, muting_percentage)
    psd_indices = psd_threshold < psd

    clean_f_hat = f_hat * psd_indices
    clean_f = np.real(np.fft.ifft(clean_f_hat))

    return clean_f


def remove_noise_fft(df: pd.DataFrame, muting_percentage: int, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    df["clean-open"] = fft(array=df.open, muting_percentage=muting_percentage)
    df["clean-high"] = fft(array=df.high, muting_percentage=muting_percentage)
    df["clean-low"] = fft(array=df.low, muting_percentage=muting_percentage)
    df["clean-close"] = fft(array=df.close, muting_percentage=muting_percentage)
    df = df.dropna()

    return df
