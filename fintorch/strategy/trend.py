from typing import List

import numpy as np
import pandas as pd
from fintorch.cross_validation.fold import Fold
from fintorch.position import Position
from tqdm import tqdm

from fintorch.strategy.strategy import Strategy


class TrendStrategy(Strategy):
    def __init__(self, show_progress_bar: bool = True):
        super().__init__(name="Trend Strategy", short_name="Trend", show_progress_bar=show_progress_bar)

    def prepare_dataframe(self, fold: Fold) -> pd.DataFrame:
        df = fold.test_set.df.copy()
        df.reset_index(inplace=True)
        df['side'] = np.where(1 == fold.best_dev_on_test_metrics.prediction, 1, -1)

        return df

    def backtest(self, fold: Fold) -> List[Position]:
        df = self.prepare_dataframe(fold)

        bar = range(len(df))
        if self.show_progress_bar:
            bar = tqdm(list(bar))
            bar.set_description_str("Backtesting Trend Strategy")

        entry_index = 0
        positions: List[Position] = []
        for index in bar:
            if df.side.iloc[entry_index] != df.side.iloc[index]:
                exit_price = df.open.iloc[index]
                position = self._create_position(df=df, entry_index=entry_index, exit_index=index,
                                                 exit_price=exit_price)
                positions.append(position)
                entry_index = index

        return positions
