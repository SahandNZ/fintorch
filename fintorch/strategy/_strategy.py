from abc import abstractmethod
from typing import Union, List

import pandas as pd

from ..component import Component
from ..deep.module import Module
from ..dtype import Position, DataCollection
from ..enum import TimeFrame
from ..exchange import Exchange, Market
from ..utils.hash import static_list_hash
from ..utils.timestamp import floor_timestamp


class Strategy(Component):
    def __init__(self, name: str, short_name: str, module: Module):
        super().__init__(name=name, short_name=short_name, description="")
        self.__module: Module = module

        self._static_hash: int = static_list_hash([self.short_name, self.module.static_hash])

        # engine related properties
        self.exchange: Union[Exchange, None] = None

        # process data
        self.__processed_df: Union[pd.DataFrame, None] = None
        self.__df: Union[pd.DataFrame, None] = None

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def symbol(self) -> str:
        return self.module.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.module.time_frame

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.module.dataset.time_frames

    @property
    def static_hash(self) -> int:
        return self._static_hash

    @property
    def future(self) -> Market:
        return self.exchange.future

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def preprocess(self, dc: DataCollection = None) -> None:
        if dc is None:
            dc = self.future.data.get_data_collection(symbols=[self.symbol], time_frames=self.time_frames)

        self.__processed_df = self.process_dc_to_df(dc=dc)

    @abstractmethod
    def process_dc_to_df(self, dc: DataCollection) -> pd.DataFrame:
        raise NotImplementedError()

    def on_new_candle(self) -> None:
        pass

    def on_opened_position(self, position: Position) -> None:
        pass

    def on_closed_position(self, position: Position) -> None:
        pass

    def open(self) -> None:
        self.module.open()

    def next(self, timestamp: int) -> None:
        current_open_timestamp = floor_timestamp(timestamp=timestamp, time_frame=self.time_frame)
        previous_open_timestamp = current_open_timestamp - int(self.time_frame)
        if previous_open_timestamp not in self.__processed_df.index:
            self.preprocess()

        self.__df = self.__processed_df[self.__processed_df.index < timestamp]

    def close(self) -> None:
        self.module.close()

    def __str__(self):
        return f"{str(self.module)} {self.short_name}"
