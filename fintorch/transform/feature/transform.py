import itertools
import os.path
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Set

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

        # temporal states
        self.__data: Data = None
        self.__none_timestamps: Set[int] = set()

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
    def data(self) -> Data:
        return self.__data

    @property
    def root_dir(self) -> str:
        data_dir = os.environ.get("DATA_ROOT", "./data")
        feature_transform_dir = os.path.join(data_dir, "transform/feature")
        root_dir = os.path.join(feature_transform_dir, str(self.__hash__()))

        return root_dir

    def fit(self, data: Data):
        self.__data = Data()
        for symbol, time_frame in itertools.product(self.symbols, self.time_frames):
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            df = df[self.features]
            self.__data[symbol, time_frame] = df

    def transform(self, timestamps: List[int], show_progress_bar: bool = False) -> Dict[int, np.array]:
        bar = timestamps
        if show_progress_bar:
            bar = tqdm(bar)
            bar.set_description_str(f"Creating {self.short_name} feature set")

        min_time_frames = min(self.time_frames)
        timestamp_to_feature = {}
        for timestamp in bar:
            backward_timestamp = int(timestamp // min_time_frames * min_time_frames)
            feature = self._pre_transform(timestamp=backward_timestamp)
            if feature is not None:
                timestamp_to_feature[timestamp] = feature

        return timestamp_to_feature

    def fit_transform(self, data: Data, timestamps: List[int], show_progress_bar: bool = False):
        self.fit(data=data)
        return self.transform(timestamps=timestamps, show_progress_bar=show_progress_bar)

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _pre_transform(self, timestamp: int) -> np.array:
        # load features from storage if its file exists
        path = os.path.join(self.root_dir, str(timestamp) + '.pkl')
        if os.path.exists(path):
            with open(path, "rb") as file:
                feature = pickle.load(file)
            return feature

        # return nan if timestamp exists in nan_timestamps
        elif timestamp in self.__none_timestamps:
            return None

        # create and save features of timestamp
        else:
            # create feature
            feature = self._transform(timestamp=timestamp)

            # append timestamp to nan_timestamps if it's nan
            if feature is None:
                self.__none_timestamps.add(timestamp)

            # save feature on storage if it's not nan
            else:
                if not os.path.exists(self.root_dir):
                    create_directory(self.root_dir)
                with open(path, "wb+") as file:
                    pickle.dump(feature, file)

            return feature

    def _transform(self, timestamp: int) -> np.array:
        atsf = []  # dimensions are (asset, time_frame, sequence, feature)
        for symbol in self.symbols:
            tsf = []
            for time_frame in self.time_frames:
                df = self.__data[symbol, time_frame]
                df = df[df.index < timestamp]
                df = df.iloc[-self.sequence_length:]

                if self.sequence_length != len(df):
                    return None

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
        name_hash = int.from_bytes(self.name.encode(), byteorder='big') % 10 ** 8
        symbols_hash = np.prod([int.from_bytes(symbol.encode(), byteorder='big') for symbol in self.symbols]) % 10 ** 8
        time_frames_hash = np.prod(self.time_frames) % 10 ** 8

        total_hash = name_hash * symbols_hash % 10 ** 8
        total_hash = total_hash * time_frames_hash % 10 ** 8
        total_hash = total_hash * self.sequence_length % 10 ** 8

        return total_hash

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "_FeatureTransform__data" in dct:
            del dct["_FeatureTransform__data"]
        if "_FeatureTransform__none_timestamps" in dct:
            del dct["_FeatureTransform__none_timestamps"]

        return dct
