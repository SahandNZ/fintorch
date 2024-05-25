from datetime import datetime

import numpy as np
import pandas as pd

from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import DataCollection


class SimpleStrategy(Strategy):
    def __init__(self, module: Module, name: str = "Simple Strategy", short_name: str = "Simple"):
        super().__init__(name=name, short_name=short_name, module=module)

    def process_dc_to_df(self, dc: DataCollection) -> pd.DataFrame:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df.insert(0, "datetime", [datetime.fromtimestamp(ts) for ts in df.index])

        y_hat_dict = self.module.predict(timestamps=df.index, dc=dc)
        df["trend"] = [np.argmax(y_hat) if not np.isnan(y_hat).max() else np.nan for y_hat in y_hat_dict.values()]
        df["side"] = np.where(1 == df.trend, 1, -1)

        return df

    def on_new_candle(self) -> None:
        if 0 < len(self.df):
            side = self.df.side.iloc[-1]
            position = self.future.trade.get_position(symbol=self.symbol)

            if position.side != side:
                if position.is_open:
                    self.future.trade.set_exit_order(position=position, percentage=100, comment="Exit")

                self.future.trade.set_entry_order(symbol=self.symbol, side=side, percentage=100, comment="Entry")
