import itertools
import math
import os
import pickle
from abc import abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd
from rich.progress import Progress

from ...component import Component
from ...dtype import Data
from ...enum import TimeFrame
from ...setting import TRANSFORM_DIR, FILE_COMPRESS_FACTOR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str, sequence_length):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__sequence_length: int = sequence_length

        self.__atsf_directory: str = os.path.join(TRANSFORM_DIR, self.short_name, "Asset-TimeFrame-Sequence-Feature")
        self.__sf_directory: str = os.path.join(TRANSFORM_DIR, self.short_name, "Sequence-Feature")
        create_directory(self.__atsf_directory)
        create_directory(self.__sf_directory)

        self.__data: Data = None

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
    def atsf_directory(self) -> str:
        return self.__atsf_directory

    @property
    @abstractmethod
    def _store_none(self) -> bool:
        raise NotImplementedError()

    def fit(self, data: Data, progress: Progress = None) -> None:
        items = list(itertools.product(data.symbols, data.time_frames))

        if progress is not None:
            desc = "[green]Fit data to {}".format(self.short_name)
            task = progress.add_task(description=desc, total=len(items))

        self.__data = Data()
        for symbol, time_frame in items:
            df = data[symbol, time_frame].copy()
            df = self._fit_dataframe(df)
            self.__data[symbol, time_frame] = df

            if progress is not None:
                progress.update(task, advance=1)

    @abstractmethod
    def _fit_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def transform(self, timestamp: int, symbols: List[str], time_frames: List[TimeFrame],
                  transform_missing: bool = True) -> Union[np.array, None]:
        return self._load_or_transform_atsf(timestamp, symbols, time_frames, transform_missing)

    def _load_or_transform_atsf(self, timestamp: int, symbols: List[str], time_frames: List[TimeFrame],
                                transform_missing: bool) -> Union[np.array, None]:
        symbols_static_hash = static_list_hash(symbols)
        time_frames_static_hash = static_list_hash(time_frames)
        static_hash = static_list_hash([symbols_static_hash, time_frames_static_hash])

        # define directory
        directory = os.path.join(self.atsf_directory, str(static_hash), str(self.sequence_length))
        create_directory(directory)

        # define file path
        file_compress_factor = FILE_COMPRESS_FACTOR * min(time_frames)
        file_name = math.floor(timestamp / file_compress_factor) * file_compress_factor
        file_path = os.path.join(directory, f"{file_name}.pkl")

        # safe load timestamp_to_sf if exist
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as file:
                    timestamp_to_atsf = pickle.load(file)
            except EOFError:
                timestamp_to_atsf = {}
        else:
            timestamp_to_atsf = {}

        # if timestamp is in timestamp_to_atsf then load atsf else transform and store it
        if timestamp in timestamp_to_atsf and (timestamp_to_atsf[timestamp] is not None or self._store_none):
            atsf = timestamp_to_atsf[timestamp]
        elif transform_missing:
            atsf = self._transform_atsf(timestamp, symbols, time_frames)
            if self._store_none or atsf is not None:
                timestamp_to_atsf[timestamp] = atsf
                with open(file_path, "wb+") as file:
                    pickle.dump(timestamp_to_atsf, file)
        else:
            atsf = None

        return atsf

    def _transform_atsf(self, timestamp: int, symbols: List[str], time_frames: List[TimeFrame]) -> Union[
        np.array, None]:
        atsf = []  # dimensions (asset, time frame, sequence, feature)
        for symbol in symbols:
            tsf = []
            for time_frame in time_frames:
                shifted_timestamp = self._shift_timestamp(timestamp=timestamp, time_frame=time_frame)
                sf = self._load_or_transform_sf(timestamp=shifted_timestamp, symbol=symbol, time_frame=time_frame)
                if sf is None:
                    return None

                tsf.append(sf)
            atsf.append(tsf)

        return np.array(atsf)

    @abstractmethod
    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        raise NotImplementedError()

    def _load_or_transform_sf(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
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

        # if timestamp is in timestamp_to_sf then load sf else transform and store it
        if timestamp in timestamp_to_sf and (timestamp_to_sf[timestamp] is not None or self._store_none):
            sf = timestamp_to_sf[timestamp]
        else:
            sf = self._transform_sf(timestamp=timestamp, symbol=symbol, time_frame=time_frame)
            if self._store_none or sf is not None:
                timestamp_to_sf[timestamp] = sf
                with open(file_path, "wb+") as file:
                    pickle.dump(timestamp_to_sf, file)

        return sf

    @abstractmethod
    def _transform_sf(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        raise NotImplementedError()

    def __str__(self):
        return self.name
