import numpy as np
import pandas as pd


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

    return df


def sftf(array: np.array, muting_percentage: int) -> np.array:
    f = array
    n = len(f)
    f_hat = np.fft.fft(f, n)

    psd = np.real(f_hat * np.conj(f_hat) / n)
    psd_threshold = np.percentile(psd, muting_percentage)
    psd_indices = psd_threshold < psd

    clean_f_hat = f_hat * psd_indices
    clean_f = np.real(np.fft.ifft(clean_f_hat))

    return clean_f
