import math
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

from .._transform import Transform
from ....dtype import DataCollection
from ....enum import TimeFrame
from ....setting import NUMPY_FEATURE_DTYPE


class FeatureTransform(Transform, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            look_back: int,
            features: List[str]
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=look_back,
            look_ahead=0
        )
        self.__features: List[str] = features

    @property
    def features(self) -> List[str]:
        return self.__features

    def _shift_timestamp(self, timestamp: int) -> int:
        return math.floor(timestamp / self.time_frame) * self.time_frame

    @abstractmethod
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_dc_to_sf(self, dc: DataCollection, timestamp: int) -> Union[np.array, None]:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)

        # backward cropping feature dataframe with timestamp and sequence length
        df = df[df.index < timestamp].iloc[-self.look_back:]
        fdf = self.transform_df(df=df)
        fdf = fdf.iloc[-self.dim_sequence:]
        fdf = fdf[self.features]

        if self.dim_sequence != len(fdf):
            return None

        ndf = self.normalize_df(df=fdf)
        sf = ndf.to_numpy().astype(dtype=NUMPY_FEATURE_DTYPE)

        return sf
