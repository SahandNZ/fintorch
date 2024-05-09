from abc import ABC
from datetime import datetime

import numpy as np
import pandas as pd

from .. import Strategy
from ...deep.module import Module
from ...utils.hash import static_list_hash


class DeepStrategy(Strategy, ABC):
    def __init__(self, name: str, short_name: str, module: Module):
        super().__init__(name=name, short_name=short_name, symbol=module.symbol, time_frame=module.time_frame)
        self.__module: Module = module

        self._static_hash: int = static_list_hash([
            self.module.static_hash,
            self.name,
        ])

    @property
    def module(self) -> Module:
        return self.__module

    def process_df(self) -> pd.DataFrame:
        dc = self.future.data.get_data_collection(symbols=[self.symbol], time_frames=[self.time_frame])
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df.insert(0, "datetime", [datetime.fromtimestamp(ts / 1000) for ts in df.index])

        # add trend (up) column
        y_hat_dict = self.module.predict(timestamps=df.index, dc=dc)
        df["trend"] = [np.argmax(y_hat) if not np.isnan(y_hat).max() else np.nan for y_hat in y_hat_dict.values()]
        df = df.dropna()
        df["side"] = np.where(1 == df.trend, 1, -1)

        return df

    def open(self) -> None:
        self.module.open()

    def close(self) -> None:
        self.module.close()

    def __str__(self):
        return f"{str(self.module)} {self.short_name}"
