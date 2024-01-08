import itertools
import os
import pickle
from abc import abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd
from rich.progress import Progress

from fintorch.component import Component
from fintorch.data import Data
from fintorch.defaults import TRANSFORM_DIR
from fintorch.utils.directory import create_directory
from fintorch.utils.hash import static_list_hash


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str, sequence_length):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__sequence_length: int = sequence_length

        self.__directory: str = os.path.join(TRANSFORM_DIR, self.short_name)
        create_directory(self.__directory)

        self.__data: Data = None

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def data(self) -> Data:
        return self.__data

    @property
    def symbols(self) -> List[str]:
        return self.__data.symbols

    @property
    def time_frames(self) -> List[int]:
        return self.__data.time_frames

    @property
    def directory(self) -> str:
        return self.__directory

    @property
    @abstractmethod
    def _save_none(self) -> bool:
        raise NotImplementedError()

    def fit(self, data: Data, progress: Progress = None) -> None:
        items = list(itertools.product(data.symbols, data.time_frames))

        if progress is not None:
            desc = "[green]Fit data to {}".format(self.short_name)
            task = progress.add_task(description=desc, total=len(items))

        self.__data = Data()
        for symbol, time_frame in items:
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            self.__data[symbol, time_frame] = df

            if progress is not None:
                progress.update(task, advance=1)

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        NotImplemented()

    def transform(self, timestamp: int) -> Union[np.array, None]:
        stv = []  # dimensions (symbol, time frame, value [sequence, feature / label])
        for symbol in self.symbols:
            tv = []
            for time_frame in self.time_frames:
                shifted_timestamp = self._shift_timestamp(timestamp=timestamp, time_frame=time_frame)
                value = self._pre_transform(timestamp=shifted_timestamp, symbol=symbol, time_frame=time_frame)
                if value is None:
                    return None

                tv.append(value)
            stv.append(tv)

        return np.array(stv)

    @abstractmethod
    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        NotImplemented()

    def _pre_transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        directory = os.path.join(self.directory, symbol, str(time_frame))
        file_path = os.path.join(directory, f"{self.sequence_length}.pkl")
        create_directory(directory)

        # load timestamp to value dict
        if os.path.exists(file_path) and 0 < os.path.getsize(file_path):
            with open(file_path, "rb") as file:
                timestamp_to_value = pickle.load(file)
        else:
            timestamp_to_value = {}

        # if timestamp is in timestamp to value then return it else calculate and store it
        if timestamp in timestamp_to_value:
            value = timestamp_to_value[timestamp]
        else:
            value = self._transform(timestamp=timestamp, symbol=symbol, time_frame=time_frame)
            if self._save_none or value is not None:
                timestamp_to_value[timestamp] = value
                with open(file_path, "wb+") as file:
                    pickle.dump(timestamp_to_value, file)

        return value

    @abstractmethod
    def _transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        NotImplemented()

    def __str__(self):
        return self.name

    def __hash__(self):
        hash_values = [
            self.name,
            static_list_hash(self.symbols),
            static_list_hash(self.time_frames),
            self.sequence_length
        ]

        return static_list_hash(hash_values)
