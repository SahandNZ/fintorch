import itertools
from abc import abstractmethod

import numpy as np
import pandas as pd
from rich.progress import Progress

from fintorch.component import Component
from fintorch.data import Data


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str):
        super().__init__(name=name, short_name=short_name, description=description)
        self.__data: Data = None

    @property
    def data(self) -> Data:
        return self.__data

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

    @abstractmethod
    def transform(self, *args) -> np.array:
        NotImplemented()

    def __str__(self):
        return self.name
