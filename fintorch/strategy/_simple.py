from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import DataCollection


class SimpleStrategy(Strategy):
    def __init__(
            self,
            module: Module,
            name: str = "Simple Strategy",
            short_name: str = "Simple",
            indicators: List[str] = None,
            height_ratios: List[int] = None
    ) -> None:
        if indicators is None:
            indicators = []
        if height_ratios is None:
            height_ratios = [2, 1]

        super().__init__(
            name=name,
            short_name=short_name,
            module=module,
            indicators=["trend"] + indicators,
            height_ratios=height_ratios
        )

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

    def _draw_indicators_plot(self, axis: List[plt.Axes], df: pd.DataFrame) -> None:
        df = df.copy()
        df["y"] = df.low.min()
        udf = df[1 == df.trend]
        ddf = df[0 == df.trend]
        ndf = df[np.isnan(df.trend)]

        size = 2 ** 11 // len(df)

        # draw scatters
        ohlc_ax = axis[0]
        ohlc_ax.scatter(x=udf.index.to_list(), y=udf.y, s=size, marker='o', c="g", label="Up Prediction")
        ohlc_ax.scatter(x=ddf.index.to_list(), y=ddf.y, s=size, marker='o', c="r", label="Down Prediction")
        ohlc_ax.scatter(x=ndf.index.to_list(), y=ndf.y, s=size, marker='o', c="gray", label="Nan Prediction")
        ohlc_ax.legend()
