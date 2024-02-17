import os
import pickle
import time
from abc import abstractmethod
from datetime import datetime
from typing import Generator, Union, List

import numpy as np
import pandas as pd

from ...component import Component
from ...dtype import DataCollection
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

        self.__processed_df: pd.DataFrame = pd.DataFrame()

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
    def processed_df(self) -> pd.DataFrame:
        return self.__processed_df

    @property
    def directory(self) -> str:
        return os.path.join(TRANSFORM_DIR, self.short_name, self.symbol, str(int(self.time_frame)))

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"sequence-length-{self.dim_sequence}.pkl")

    def load_sf(self, timestamps: List[int]) -> Generator:
        # safe load timestamp_to_sf
        try:
            with open(self.path, "rb") as file:
                timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError):
            timestamp_to_sf = {}

        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            sf = timestamp_to_sf.get(shifted_timestamp, None)
            yield sf

    def transform_sf(self, dc: DataCollection, timestamps: List[int]) -> Generator:
        # safe load timestamp_to_sf
        try:
            with open(self.path, "rb") as file:
                timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError):
            timestamp_to_sf = {}

        # update sf values if needed
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)

            # load or transform sf
            sf = timestamp_to_sf.get(shifted_timestamp, None)
            if sf is None and self._can_not_be_none(dc=dc, timestamp=shifted_timestamp):
                sf = self.__transform_sf(dc=dc, timestamp=shifted_timestamp)
                timestamp_to_sf[shifted_timestamp] = sf

            yield sf

        # dump timestamp_to_sf
        create_directory(self.directory)
        with open(self.path, "wb+") as file:
            pickle.dump(timestamp_to_sf, file)

    def __transform_sf(self, dc: DataCollection, timestamp: int) -> Union[np.array, None]:
        df = self.__transform_dc_to_df(dc=dc, timestamp=timestamp)
        sf = self._transform_df_to_sf(df=df, timestamp=timestamp)

        return sf

    def __transform_dc_to_df(self, dc: DataCollection, timestamp: int) -> pd.DataFrame:
        if 0 == len(self.processed_df) or timestamp not in self.processed_df.index:
            df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
            self.__processed_df = self._process_df(df.copy())

        return self.processed_df

    @abstractmethod
    def _shift_timestamp(self, timestamp: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _can_not_be_none(self, dc: DataCollection, timestamp: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _transform_df_to_sf(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "{} {} {}".format(self.symbol, self.time_frame, self.short_name)
