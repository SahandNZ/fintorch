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

        self.__symbols: List[str] = None
        self.__time_frames: List[int] = None
        self.__root_directory: str = None
        self.__stv_directory: str = None
        self.__v_directory: str = None

        self.__data: Data = None

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def data(self) -> Data:
        return self.__data

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[int]:
        return self.__time_frames

    @property
    def stv_directory(self) -> str:
        return self.__stv_directory

    @property
    def v_directory(self) -> str:
        return self.__v_directory

    def fit(self, data: Data, progress: Progress = None) -> None:
        self._set_private_properties(data=data)
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

    def _set_private_properties(self, data: Data):
        self.__symbols = data.symbols
        self.__time_frames = data.time_frames
        self.__root_directory = os.path.join(TRANSFORM_DIR, self.short_name, str(self.__hash__()))
        self.__stv_directory = os.path.join(self.__root_directory, "symbol-timeframe-value")
        self.__v_directory = os.path.join(self.__root_directory, "value")
        create_directory(self.__stv_directory)
        create_directory(self.__v_directory)

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        NotImplemented()

    def transform(self, timestamp: int) -> Union[np.array, None]:
        file_path = os.path.join(self.stv_directory, f"{timestamp}.pkl")
        if os.path.exists(file_path) and 0 < os.path.getsize(file_path):
            with open(file_path, "rb") as file:
                result = pickle.load(file)

        else:
            result = self._iterate_transforms(timestamp=timestamp)
            with open(file_path, "wb+") as file:
                pickle.dump(result, file)

        return result

    def _iterate_transforms(self, timestamp: int) -> Union[np.array, None]:
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
        file_path = os.path.join(self.v_directory, f"{timestamp}.pkl")
        if os.path.exists(file_path) and 0 < os.path.getsize(file_path):
            with open(file_path, "rb") as file:
                result = pickle.load(file)

        else:
            result = self._transform(timestamp=timestamp, symbol=symbol, time_frame=time_frame)
            with open(file_path, "wb+") as file:
                pickle.dump(result, file)

        return result

    @abstractmethod
    def _transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        NotImplemented()

    def __str__(self):
        return self.name

    def __hash__(self):
        hash_values = [static_list_hash(self.symbols), static_list_hash(self.time_frames), self.sequence_length]
        total_hash = static_list_hash(hash_values)
        return total_hash
