import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from .._transform import Transform
from ....dtype import DataCollection
from ....enum import TimeFrame
from ....setting import NUMPY_FEATURE_DTYPE


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, symbol: str, time_frame: TimeFrame, dim_sequence: int,
                 look_back: int, features: List[str]):
        super().__init__(name=name, short_name=short_name, description="", symbol=symbol, time_frame=time_frame,
                         dim_sequence=dim_sequence)
        self.__look_back: int = look_back
        self.__features: List[str] = features

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def features(self) -> List[str]:
        return self.__features

    def _shift_timestamp(self, timestamp: int) -> int:
        return math.floor(timestamp / self.time_frame) * self.time_frame

    def _can_not_be_none(self, dc: DataCollection, timestamp: int) -> bool:
        symbol_info = dc.get_symbol_info(symbol=self.symbol)
        first_timestamp = symbol_info.on_board_timestamp + int(self.time_frame) * self.look_back * 2
        last_timestamp = datetime.now().timestamp() // int(self.time_frame) * int(self.time_frame)

        return first_timestamp <= timestamp <= last_timestamp

    @abstractmethod
    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_df_to_sf(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        # backward cropping feature dataframe with timestamp and sequence length
        fdf = df[df.index < timestamp]
        fdf = fdf.iloc[-self.dim_sequence:]
        fdf = fdf[self.features]

        if self.dim_sequence != len(fdf):
            return None

        # z-score standardization and min-max normalization (keep negative values)
        zdf = (fdf - fdf.mean()) / fdf.std()
        ndf = zdf / (zdf.max() - zdf.min())
        sf = ndf.to_numpy().astype(dtype=NUMPY_FEATURE_DTYPE)

        return sf
