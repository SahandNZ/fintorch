import numpy as np
import pandas as pd

from fintorch.transform.transform import Transform


class DfftTransform(Transform):
    def __init__(self, muting_percentage: int):
        super().__init__(name="Discrete Fast Fourier Transform", short_name="DFFT", description="")
        self.__muting_percentage: str = muting_percentage

    @property
    def muting_percentage(self) -> int:
        return self.__muting_percentage

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform(self, array: np.array) -> np.array:
        f = array
        n = len(f)
        f_hat = np.fft.fft(f, n)

        psd = np.real(f_hat * np.conj(f_hat) / n)
        psd_threshold = np.percentile(psd, self.muting_percentage)
        psd_indices = psd_threshold < psd

        clean_f_hat = f_hat * psd_indices
        clean_f = np.real(np.fft.ifft(clean_f_hat))

        return clean_f

    def fit_transform(self, *args):
        pass
