from typing import List

import numpy as np
import pandas as pd
from fintorch.cross_validation.fold import Fold
from fintorch.position import Position
from tqdm import tqdm

from fintorch.strategy.strategy import Strategy


class StaticTpSlTrendStrategy(Strategy):
    def __init__(self, tp_rate: float, sl_rate: float, show_progress_bar: bool = True):
        super().__init__(name="Static-TP-SL-Trend Strategy", short_name="Static-TP-SL-Trend",
                         show_progress_bar=show_progress_bar)
        self.__tp_rate: float = tp_rate
        self.__sl_rate: float = sl_rate

    @property
    def tp_rate(self) -> float:
        return self.__tp_rate

    @property
    def sl_rate(self) -> float:
        return self.__sl_rate

    def prepare_dataframe(self, fold: Fold) -> pd.DataFrame:
        df = fold.test_set.df.copy()
        df.reset_index(inplace=True)
        df['side'] = np.where(1 == fold.best_dev_on_test_metrics.prediction, 1, -1)
        df['toggle'] = df.side != df.side.shift(1)
        df = df.dropna()

        return df

    def backtest(self, fold: Fold) -> List[Position]:
        df = self.prepare_dataframe(fold)

        bar = range(len(df))
        if self.show_progress_bar:
            bar = tqdm(list(bar))
            bar.set_description_str("Backtesting Static-TP-SL-Trend Strategy")

        entry_index = None
        positions: List[Position] = []
        for index in bar:
            if entry_index is None and df.toglle.iloc[index]:
                entry_index = index
                side = df.side.iloc[index]
                sl_price = df.open.iloc[index] * (1 - side * self.sl_rate)
                tp_price = df.open.iloc[index] * (1 + side * self.tp_rate)

            if entry_index is not None:
                p = None
                if df.low.iloc[index] <= sl_price <= df.high.iloc[index]:
                    p = self._create_position(df, entry_index=entry_index, exit_index=index, exit_price=sl_price)
                elif df.low.iloc[index] <= tp_price <= df.high.iloc[index]:
                    p = self._create_position(df, entry_index=entry_index, exit_index=index, exit_price=tp_price)

                if p is not None:
                    positions.append(p)
                    entry_index = None

        return positions
