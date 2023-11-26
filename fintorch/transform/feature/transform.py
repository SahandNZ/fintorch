import itertools
import os.path
import pickle
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from fintorch.data import Data
from fintorch.transform.transform import Transform
from fintorch.utils.directory import create_directory


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbols: List[str], time_frames: int,
                 sequence_length: int, features: List[str]):
        super().__init__(name, short_name, description)
        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__sequence_length: int = sequence_length
        self.__features: List[str] = features

        self.__root: str = None

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

    @property
    def root(self) -> str:
        if self.__root is None:
            data_root = os.environ.get("DATA_ROOT", "./data")
            feature_transform_root = os.path.join(data_root, "transform/feature")
            self.__root = os.path.join(feature_transform_root, str(self.__hash__()))

        return self.__root

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
            bar.set_description_str(f"Creating {self.short_name} feature set")

        features = []
        for timestamp in bar:
            feature = self._pre_transform(data=data, timestamp=timestamp)
            features.append(feature)
        return features

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _pre_transform(self, data: Data, timestamp: int) -> np.array:
        path = os.path.join(self.root, str(timestamp) + '.pkl')
        # load features of specified timestamp
        if os.path.exists(path):
            with open(path, "rb") as file:
                sample = pickle.load(file)
            return sample

        # create and save features of specified timestamp
        else:
            sample = self._transform(data=data, timestamp=timestamp)
            if not os.path.exists(self.root):
                create_directory(self.root)
            with open(path, "wb+") as file:
                pickle.dump(sample, file)
            return sample

    def _transform(self, data: Data, timestamp: int) -> np.array:
        atsf = []  # dimensions are (asset, time_frame, sequence, feature)
        for symbol in self.symbols:
            tsf = []
            for time_frame in self.time_frames:
                df = data[symbol, time_frame]
                df = df[df.index < timestamp]
                df = df.iloc[-self.sequence_length:]

                if self.sequence_length != len(df):
                    return np.nan

                df = df / df.std()
                sf = df.to_numpy()

                tsf.append(sf)
            atsf.append(tsf)

        return np.array(atsf)

    def __eq__(self, other):
        is_symbols_equal = self.symbols == other.symbols
        is_time_frames_equal = self.time_frames == other.time_frames
        is_sequence_lengths_equal = self.sequence_length == other.sequence_length

        return super().__eq__(other) and is_symbols_equal and is_time_frames_equal and is_sequence_lengths_equal

    def __hash__(self):
        symbols_hash = np.prod([int.from_bytes(symbol.encode(), byteorder='big') for symbol in self.symbols]) % 10 ** 9
        time_frames_hash = np.prod(self.time_frames) % 10 ** 9
        total_hash = symbols_hash * time_frames_hash * self.sequence_length % 10 ** 9

        return total_hash
