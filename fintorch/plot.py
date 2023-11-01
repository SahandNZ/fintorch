import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def draw_ohlcv_plot(df: pd.DataFrame, volume: bool = True, warn_too_much_data: int = 1000):
    pdf = df.copy()
    pdf.index = pd.DatetimeIndex(pdf['datetime'])
    pdf = pdf[['open', 'high', 'low', 'close', 'volume']]
    fig, axs = mpf.plot(pdf, type='candle', volume=volume, figsize=(20, 10), style='binance', returnfig=True,
                        warn_too_much_data=warn_too_much_data)
    ohlc_ax = axs[0]
    volume_ax = axs[1]
    return fig, ohlc_ax, volume_ax


def draw_rectangle(ax: plt.Axes, x1: int, y1: int, x2: int, y2: int, color: str = None, alpha: float = 0.25):
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    color = color if color else 'g' if y1 < y2 else 'r'
    area = Rectangle((x, y), width, height, color=color, alpha=alpha)
    ax.add_patch(area)


def draw_trend(ax: plt.Axes, df: pd.DataFrame):
    label, start_index = -1, 0
    for index in range(len(df)):
        if label != df.label.iloc[index] or index == len(df) - 1:
            if 0 <= label:
                # label
                minimum_price = df.low.iloc[start_index: index].min()
                maximum_price = df.high.iloc[start_index: index].max()
                y1 = minimum_price if 1 == label else maximum_price
                y2 = maximum_price if 1 == label else minimum_price
                draw_rectangle(ax=ax, x1=start_index, y1=y1, x2=index, y2=y2)

                # prediction
                for i in range(start_index, index):
                    if not np.isnan(df.prediction.iloc[i]):
                        color = "r" if 0 == df.prediction.iloc[i] else "g"
                        ax.scatter(i, minimum_price, s=8, marker='o', c=color)

            label = df.label.iloc[index]
            start_index = index
