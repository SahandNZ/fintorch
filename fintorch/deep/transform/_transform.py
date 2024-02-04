import os
import pickle
from abc import abstractmethod
from datetime import datetime
from typing import Generator, Union, List

import numpy as np
import pandas as pd

from ...component import Component
from ...dtype import Data
from ...enum import TimeFrame
from ...setting import TRANSFORM_DIR
from ...utils.directory import create_directory


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: TimeFrame,
                 dim_sequence: int):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__symbol: str = symbol
        self.__time_frame: TimeFrame = time_frame
        self.__dim_sequence: int = dim_sequence

        self.__preprocessed_dataframe: pd.DataFrame = None

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.__time_frame

    @property
    def dim_sequence(self) -> int:
        return self.__dim_sequence

    @property
    def processed_dataframe(self) -> pd.DataFrame:
        return self.__preprocessed_dataframe

    @property
    def directory(self) -> str:
        return os.path.join(TRANSFORM_DIR, self.short_name, self.symbol, str(int(self.time_frame)))

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"sequence-length-{self.dim_sequence}.pkl")

    def load_sf(self, timestamps: List[int]) -> List[Union[np.array, None]]:
        # load timestamp_to_sf
        try:
            with open(self.path, "rb") as file:
                timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError) as e:
            timestamp_to_sf = {}

        sf_values = []
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            sf = timestamp_to_sf.get(shifted_timestamp, None)
            sf_values.append(sf)

        return sf_values

    def transform_sf(self, data: Data, timestamps: List[int]) -> Generator[Union[np.array, None], None, None]:
        # load timestamp_to_sf
        try:
            with open(self.path, "rb") as file:
                timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError) as e:
            timestamp_to_sf = {}

        # update sf values if needed
        overwrite_file = False
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)

            # load sf value and update it if needed
            if timestamp in timestamp_to_sf:
                sf = timestamp_to_sf[shifted_timestamp]
                if sf is None and not self._can_be_none(timestamp=shifted_timestamp):
                    sf = self._transform_sf(data=data, timestamp=shifted_timestamp)
                    timestamp_to_sf[shifted_timestamp] = sf
                    overwrite_file = True
            else:
                sf = self._transform_sf(data=data, timestamp=shifted_timestamp)
                timestamp_to_sf[shifted_timestamp] = sf
                overwrite_file = True

            yield sf

        # dump timestamp_to_sf if needed
        create_directory(self.directory)
        if overwrite_file:
            with open(self.path, "wb+") as file:
                pickle.dump(timestamp_to_sf, file)

    def _transform_sf(self, data: Data, timestamp: int) -> Union[np.array, None]:
        df = self._preprocess_sf(data=data, timestamp=timestamp)
        sf = self._transform_dataframe(df=df, timestamp=timestamp)

        return sf

    def _preprocess_sf(self, data: Data, timestamp: int) -> pd.DataFrame:
        if self.processed_dataframe is None or timestamp not in self.processed_dataframe.index:
            current_open_timestamp = datetime.now().timestamp() // int(self.time_frame) * int(self.time_frame)
            if timestamp not in data[self.symbol, self.time_frame].index and timestamp <= current_open_timestamp:
                raise ValueError("Data has missing value at in {}-{} at {}."
                                 .format(self.symbol, self.time_frame, datetime.fromtimestamp(timestamp)))

            df = data[self.symbol, self.time_frame].copy()
            self.__preprocessed_dataframe = self._preprocess_dataframe(df)

        return self.__preprocessed_dataframe

    @abstractmethod
    def _shift_timestamp(self, timestamp: int) -> int:
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
        return "{} {} {}".format(self.symbol, self.time_frame, self.short_name)
