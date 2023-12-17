import itertools
import os.path
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from rich.progress import Progress

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
    def directory(self) -> str:
        data_dir = os.environ.get("DATA_ROOT", "./data")
        feature_transform_dir = os.path.join(data_dir, "transform/feature")
        symbols_str = ", ".join([str(symbol) for symbol in self.symbols])
        time_frames_str = ", ".join([str(time_frame) for time_frame in self.time_frames])
        directory = os.path.join(feature_transform_dir, self.short_name, symbols_str, time_frames_str)
        create_directory(directory)

        return directory

    def fit(self, data: Data):
        self.__data = Data()
        for symbol, time_frame in itertools.product(self.symbols, self.time_frames):
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            df = df[self.features]
            self.__data[symbol, time_frame] = df

    def transform(self, timestamps: List[int], progress: Progress = None) -> Dict[int, np.array]:
        if progress is not None:
            symbols_str = ", ".join([symbol for symbol in self.symbols])
            time_frames_str = ", ".join([str(time_frame) for time_frame in self.time_frames])
            desc = "[green]Creating ({} - {} - {}) features".format(symbols_str, time_frames_str, self.short_name)
            task = progress.add_task(description=desc, total=len(timestamps))

        min_time_frames = min(self.time_frames)
        features_dict = {}
        for timestamp in timestamps:
            backward_timestamp = int(timestamp // min_time_frames * min_time_frames)
            feature = self._pre_transform(timestamp=backward_timestamp)
            if feature is not None:
                features_dict[timestamp] = feature

            if progress is not None:
                progress.advance(task, advance=1)

        return features_dict

    def fit_transform(self, data: Data, timestamps: List[int], progress: Progress = None):
        self.fit(data=data)
        return self.transform(timestamps=timestamps, progress=progress)

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _pre_transform(self, timestamp: int) -> np.array:
        # load features from storage if its file exists and it doesn't empty
        file_path = os.path.join(self.directory, str(timestamp) + '.pkl')
        if os.path.exists(file_path) and 0 < os.path.getsize(file_path):
            with open(file_path, "rb") as file:
                feature = pickle.load(file)
            return feature

        # return none if timestamp exists in nan_timestamps
        elif timestamp in self.__none_timestamps:
            return None

        # create and save features of timestamp
        else:
            # create feature
            feature = self._transform(timestamp=timestamp)

            # append timestamp to none_timestamps if it's None
            if feature is None:
                self.__none_timestamps.add(timestamp)

            # save feature on storage if it's not None
            else:
                with open(file_path, "wb+") as file:
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

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "_FeatureTransform__data" in dct:
            del dct["_FeatureTransform__data"]
        if "_FeatureTransform__none_timestamps" in dct:
            del dct["_FeatureTransform__none_timestamps"]

        return dct
