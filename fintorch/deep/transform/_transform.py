import os
import pickle
import time
from abc import abstractmethod
from datetime import datetime
from typing import Generator, Union, List

import numpy as np
import pandas as pd
from rich.progress import Progress

from ...component import Component
from ...dtype import DataCollection, SymbolInfo
from ...enum import TimeFrame
from ...setting import TRANSFORM_DIR
from ...utils.directory import create_directory


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
    def look_back(self) -> int:
        return self.__look_back

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def processed_df(self) -> pd.DataFrame:
        return self.__processed_df

    @property
    def directory(self) -> str:
        return os.path.join(TRANSFORM_DIR, self.short_name, self.symbol, str(int(self.time_frame)))

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"sequence-length-{self.dim_sequence}.pkl")

    def get_start_timestamp(self, dc: DataCollection) -> int:
        # create on board timestamp
        symbol_info = dc.get_symbol_info(symbol=self.symbol)
        on_board_datetime = symbol_info.on_board_datetime
        on_board_timestamp = on_board_datetime.replace(month=(on_board_datetime.month + 1) % 12, day=1).timestamp()
        on_board_timestamp = on_board_timestamp // int(self.time_frame) * int(self.time_frame)

        # create first available timestamp (seems there is a issue in binance candles database)
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        df_first_timestamp = df.index[0]

        return int(max(on_board_timestamp, df_first_timestamp) + self.look_back * int(self.time_frame))

    def get_valid_timestamps(self, dc: DataCollection) -> List[int]:
        current_timestamp = datetime.now().timestamp() // int(self.time_frame) * int(self.time_frame)

        start_timestamp = int(self.get_start_timestamp(dc=dc) + self.look_back * int(self.time_frame))
        stop_timestamp = int(current_timestamp - self.look_ahead * int(self.time_frame))
        timestamps = list(range(start_timestamp, stop_timestamp, int(self.time_frame)))

        return timestamps

    def prepare_sf(self, dc: DataCollection, progress: Union[Progress, None] = None) -> None:
        timestamps = self.get_valid_timestamps(dc=dc)

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
        return "{} - {} - {}".format(self.short_name, self.symbol, str(self.time_frame))
