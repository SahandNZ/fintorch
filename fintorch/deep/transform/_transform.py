import os
import pickle
from abc import abstractmethod
from typing import Dict, Generator, Union, List

import numpy as np
import pandas as pd
from rich.progress import Progress

from ...component import Component
from ...dtype import DataCollection
from ...enum import TimeFrame
from ...setting import TRANSFORM_DIR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash


class Transform(Component):
    def __init__(
            self,
            name: str,
            short_name: str,
            description: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            look_back: int,
            look_ahead: int
    ):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__symbol: str = symbol
        self.__time_frame: TimeFrame = time_frame if isinstance(time_frame, TimeFrame) else TimeFrame(time_frame)
        self.__dim_sequence: int = dim_sequence
        self.__look_back: int = look_back
        self.__look_ahead: int = look_ahead

        self.__timestamp_to_sf: Dict[int, List] = {}
        self.__processed_df: pd.DataFrame = pd.DataFrame()

        self.__static_hash: int = static_list_hash([
            self.short_name,
            self.symbol,
            self.time_frame,
            self.look_back,
            self.look_ahead
        ])

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
    def look_back(self) -> int:
        return self.__look_back

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def timestamp_to_sf(self) -> Dict[int, List]:
        return self.__timestamp_to_sf

    @property
    def processed_df(self) -> pd.DataFrame:
        return self.__processed_df

    @property
    def directory(self) -> str:
        return os.path.join(TRANSFORM_DIR, self.short_name, self.symbol, str(int(self.time_frame)))

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"sequence-length-{self.dim_sequence}.pkl")

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    def open(self) -> None:
        try:
            with open(self.path, "rb") as file:
                self.__timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__timestamp_to_sf = {}

    def close(self) -> None:
        create_directory(self.directory)
        with open(self.path, "wb+") as file:
            pickle.dump(self.timestamp_to_sf, file)

    def get_start_timestamp(self, dc: DataCollection) -> int:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        start_timestamp = df.index[0]
        start_timestamp = int(start_timestamp + self.look_back * int(self.time_frame))

        return start_timestamp

    def get_stop_timestamp(self, dc: DataCollection) -> int:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        stop_timestamp = df.index[-1]
        stop_timestamp = int(stop_timestamp - self.look_ahead * int(self.time_frame)) + int(self.time_frame)

        return stop_timestamp

    def get_timestamps(self, dc: DataCollection) -> List[int]:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)
        timestamps = list(range(start_timestamp, stop_timestamp, int(self.time_frame)))

        return timestamps

    def prepare_sf(
            self, dc: DataCollection,
            timestamps: Union[List[int], None] = None,
            progress: Union[Progress, None] = None
    ) -> None:
        if timestamps is None:
            timestamps = self.get_timestamps(dc=dc)

        # create rich progress bar
        if progress is not None:
            description = "Creating SF values of {}".format(str(self))
            task = progress.add_task(description=description, total=len(timestamps))
        else:
            task = None

        # create sf values
        for _ in self.transform_sf(dc=dc, timestamps=timestamps):
            if progress is not None:
                progress.update(task_id=task, advance=1)

        # hide rich progress bar
        if progress is not None:
            progress.update(task_id=task, visible=False)

    def load_sf(self, timestamps: List[int]) -> Generator[Union[np.array, None], None, None]:
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            sf = self.timestamp_to_sf.get(shifted_timestamp, None)

            yield sf

    def transform_sf(self, dc: DataCollection, timestamps: List[int]) -> Generator[np.array, None, None]:
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            sf = self.timestamp_to_sf.get(shifted_timestamp, None)
            if not self.__is_sf_valid(dc=dc, timestamp=shifted_timestamp, sf=sf):
                sf = self.__transform_sf(dc=dc, timestamp=shifted_timestamp)
                self.timestamp_to_sf[shifted_timestamp] = sf

            yield sf

    def __is_sf_valid(self, dc: DataCollection, timestamp: int, sf: Union[np.array, None]) -> bool:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)
        is_timestamp_valid = start_timestamp <= timestamp < stop_timestamp
        is_sf_valid = is_timestamp_valid and sf is not None

        return is_sf_valid

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
    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _transform_df_to_sf(self, df: pd.DataFrame, timestamp: int) -> Union[List, None]:
        raise NotImplementedError()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        return "{} - {} - {}".format(self.short_name, self.symbol, str(self.time_frame))
