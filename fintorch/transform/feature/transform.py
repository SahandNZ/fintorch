import itertools
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from fintorch.data import Data
from fintorch.transform.transform import Transform


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbols: List[str], time_frames: int,
                 sequence_length: int, features: List[str]):
        super().__init__(name, short_name, description)
        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__sequence_length: int = sequence_length
        self.__features: List[str] = features

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[int]:
        return self.__time_frames

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def features(self) -> List[str]:
        return self.__features

    def fit(self, data: Data):
        feature_data = Data()
        for symbol, time_frame in itertools.product(self.symbols, self.time_frames):
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            df = df[self.features]
            feature_data[symbol, time_frame] = df

        return feature_data

    def transform(self, data: Data, timestamps: List[int], show_progress_bar: bool = False) -> List[np.array]:
        bar = timestamps
        if show_progress_bar:
            bar = tqdm(bar)
            bar.set_description_str("Creating feature set")

        features = []
        for timestamp in bar:
            feature = self._transform(data=data, timestamp=timestamp)
            features.append(feature)
        return features

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _transform(self, data: Data, timestamp: int) -> np.array:
        atsf = []
        for symbol in self.symbols:
            tsf = []
            for time_frame in self.time_frames:
                df = data[symbol, time_frame]
                stop_index = int((timestamp - df.index.to_list()[0]) / time_frame)
                start_index = stop_index - self.sequence_length
                if start_index < 0:
                    return np.nan

                df = df.iloc[start_index: stop_index]
                df = df / df.std()
                sf = df.to_numpy()

                tsf.append(sf)
            atsf.append(tsf)

        return np.array(atsf)
