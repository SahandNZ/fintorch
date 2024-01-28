import math
import os
import pickle
from abc import abstractmethod
from datetime import datetime
from typing import Union, List

import numpy as np
import pandas as pd

from ...component import Component
from ...dtype import Data
from ...enum import TimeFrame
from ...setting import TRANSFORM_DIR, FILE_COMPRESS_FACTOR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str, sequence_length: int):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__sequence_length: int = sequence_length

        self.__stsf_directory: str = os.path.join(TRANSFORM_DIR, self.short_name, "symbol-timeframe-sequence-feature")
        self.__sf_directory: str = os.path.join(TRANSFORM_DIR, self.short_name, "sequence-feature")
        self.__data: Data = Data()

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def data(self) -> Data:
        return self.__data

    @property
    def sf_directory(self) -> str:
        return self.__sf_directory

    @property
    def stsf_directory(self) -> str:
        return self.__stsf_directory

    def load(self, timestamp: int, symbols: List[str], time_frames: List[TimeFrame]) -> Union[np.array, None]:
        return self._load_or_transform_stsf(data=None, timestamp=timestamp, symbols=symbols, time_frames=time_frames,
                                            transform_missing=False)

    def load_or_transform(self, data: Union[Data, None], timestamp: int, symbols: List[str],
                          time_frames: List[TimeFrame]) -> Union[np.array, None]:
        return self._load_or_transform_stsf(data=data, timestamp=timestamp, symbols=symbols, time_frames=time_frames,
                                            transform_missing=True)

    def _load_or_transform_stsf(self, data: Union[Data, None], timestamp: int, symbols: List[str],
                                time_frames: List[TimeFrame], transform_missing: bool) -> Union[np.array, None]:
        symbols_static_hash = static_list_hash(symbols)
        time_frames_static_hash = static_list_hash(time_frames)
        static_hash = static_list_hash([symbols_static_hash, time_frames_static_hash])

        # define directory
        directory = os.path.join(self.stsf_directory, str(static_hash), str(self.sequence_length))
        create_directory(directory)

        # define file path
        file_compress_factor = FILE_COMPRESS_FACTOR // 8 * min(time_frames)
        file_name = math.floor(timestamp / file_compress_factor) * file_compress_factor
        file_path = os.path.join(directory, f"{file_name}.pkl")

        # safe load timestamp_to_stsf if exist
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as file:
                    timestamp_to_stsf = pickle.load(file)
            except EOFError:
                timestamp_to_stsf = {}
        else:
            timestamp_to_stsf = {}

        # update stsf value if needed
        if timestamp in timestamp_to_stsf:
            stsf = timestamp_to_stsf[timestamp]
            if stsf is None and not self._can_be_none(timestamp=timestamp):
                del timestamp_to_stsf[timestamp]
                stsf = self._transform_stsf(data=data, timestamp=timestamp, symbols=symbols, time_frames=time_frames)
        elif transform_missing:
            stsf = self._transform_stsf(data=data, timestamp=timestamp, symbols=symbols, time_frames=time_frames)
        else:
            stsf = None

        # store stsf if its value missed or changed
        if timestamp not in timestamp_to_stsf:
            timestamp_to_stsf[timestamp] = stsf
            with open(file_path, "wb+") as file:
                pickle.dump(timestamp_to_stsf, file)

        return stsf

    def _transform_stsf(self, data: Data, timestamp: int, symbols: List[str], time_frames: List[TimeFrame]) \
            -> Union[np.array, None]:
        stsf = []  # dimensions (symbol, time frame, sequence, feature)
        for symbol in symbols:
            tsf = []
            for time_frame in time_frames:
                timestamp = self._shift_timestamp(timestamp=timestamp, time_frame=time_frame)
                sf = self._load_or_transform_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)
                if sf is None:
                    return None

                tsf.append(sf)
            stsf.append(tsf)

        return np.array(stsf)

    def _load_or_transform_sf(self, data: Data, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        directory = os.path.join(self.sf_directory, symbol, str(time_frame), str(self.sequence_length))
        create_directory(directory)

        file_compress_factor = FILE_COMPRESS_FACTOR * time_frame
        file_name = math.floor(timestamp / file_compress_factor) * file_compress_factor
        file_path = os.path.join(directory, f"{file_name}.pkl")

        # safe load timestamp_to_sf if exist
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as file:
                    timestamp_to_sf = pickle.load(file)
            except EOFError:
                timestamp_to_sf = {}
        else:
            timestamp_to_sf = {}

        # update value of sf if needed
        if timestamp in timestamp_to_sf:
            sf = timestamp_to_sf[timestamp]
            if sf is None and not self._can_be_none(timestamp=timestamp):
                del timestamp_to_sf[timestamp]
                sf = self._transform_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)
        else:
            sf = self._transform_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)

        # store sf if its value missed or changed
        if timestamp not in timestamp_to_sf:
            timestamp_to_sf[timestamp] = sf
            with open(file_path, "wb+") as file:
                pickle.dump(timestamp_to_sf, file)

        return sf

    def _transform_sf(self, data: Data, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        df = self._preprocess_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)
        sf = self._transform_dataframe(df=df, timestamp=timestamp)

        return sf

    def _preprocess_sf(self, data: Data, timestamp: int, symbol: str, time_frame: int) -> pd.DataFrame:
        if not self.data.has(symbol, time_frame) or timestamp not in self.data[symbol, time_frame].index:
            current_open_timestamp = datetime.now().timestamp() // time_frame * time_frame
            if timestamp not in data[symbol, time_frame].index and timestamp <= current_open_timestamp:
                raise ValueError("Data has missing value at {}.".format(datetime.fromtimestamp(timestamp)))

            # TODO set dynamic value instead of 256 in below line
            start_timestamp = timestamp - time_frame * 256
            df = data[symbol, time_frame]
            df = df[start_timestamp <= df.index].copy()
            df = self._preprocess_dataframe(df)
            self.data[symbol, time_frame] = df

        return self.data[symbol, time_frame]

    @abstractmethod
    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _can_be_none(self, timestamp: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _transform_dataframe(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.name
