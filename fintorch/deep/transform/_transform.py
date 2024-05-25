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
from ...settings import TRANSFORM_DIR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash
from ...utils.timestamp import ceil_timestamp, floor_timestamp


class Transform(Component):
    def __init__(
            self,
            name: str,
            short_name: str,
            description: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            dim_feature: int,
            look_back: int,
            look_ahead: int
    ):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__symbol: str = symbol
        self.__time_frame: TimeFrame = time_frame
        self.__dim_sequence: int = dim_sequence
        self.__dim_feature: int = dim_feature
        self.__look_back: int = look_back
        self.__look_ahead: int = look_ahead

        self.__timestamp_to_sf: Dict[int, np.ndarray] = {}

        self.__static_hash: int = static_list_hash([
            self.short_name,
            self.symbol,
            int(self.time_frame * 1000),
            self.dim_sequence,
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
    def dim_feature(self) -> int:
        return self.__dim_feature

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def timestamp_to_sf(self) -> Dict[int, np.ndarray]:
        return self.__timestamp_to_sf

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    @property
    def path(self) -> str:
        return os.path.join(TRANSFORM_DIR, f"{str(self.static_hash)}.pkl")

    def open(self) -> None:
        try:
            with open(self.path, "rb") as file:
                self.__timestamp_to_sf = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__timestamp_to_sf = {}

    def close(self) -> None:
        create_directory(TRANSFORM_DIR)
        with open(self.path, "wb+") as file:
            pickle.dump(self.timestamp_to_sf, file)

    def get_start_timestamp(self, dc: DataCollection) -> int:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        start_timestamp = df.index[0]
        start_timestamp = start_timestamp + self.look_back * int(self.time_frame)
        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)

        return start_timestamp

    def get_stop_timestamp(self, dc: DataCollection) -> int:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        stop_timestamp = df.index[-1]
        stop_timestamp = stop_timestamp - self.look_ahead * int(self.time_frame)
        stop_timestamp = ceil_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)

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

    def load_sf(self, timestamps: List[int]) -> np.ndarray:
        sf_values = []
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            default_nan_sf = np.zeros((self.dim_sequence, self.dim_feature)) * np.nan
            sf = self.timestamp_to_sf.get(shifted_timestamp, default_nan_sf)
            sf_values.append(sf)

        sf_values = np.array(sf_values)

        return sf_values

    def transform_sf(self, dc: DataCollection, timestamps: List[int]) -> Generator[np.ndarray, None, None]:
        for timestamp in timestamps:
            shifted_timestamp = self._shift_timestamp(timestamp=timestamp)
            sf = self.timestamp_to_sf.get(shifted_timestamp, np.zeros((self.dim_sequence, self.dim_feature)) * np.nan)
            if not self._is_sf_valid(dc=dc, timestamp=shifted_timestamp, sf=sf):
                sf = self._transform_dc_to_sf(dc=dc, timestamp=shifted_timestamp)
                self.timestamp_to_sf[shifted_timestamp] = sf

            yield sf

    @abstractmethod
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _shift_timestamp(self, timestamp: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _transform_dc_to_sf(self, dc: DataCollection, timestamp: int) -> np.ndarray:
        raise NotImplementedError()

    def _is_sf_valid(self, dc: DataCollection, timestamp: int, sf: np.ndarray) -> bool:
        start_timestamp = self.get_start_timestamp(dc=dc)
        stop_timestamp = self.get_stop_timestamp(dc=dc)
        is_timestamp_valid = start_timestamp <= timestamp < stop_timestamp
        is_value_valid = not np.isnan(sf).max()
        is_valid = is_timestamp_valid and is_value_valid

        return is_valid

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self) -> str:
        return "{} - {} - {}".format(self.short_name, self.symbol, str(self.time_frame))
