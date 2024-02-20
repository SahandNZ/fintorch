from datetime import datetime

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.patches import Rectangle

from fintorch.dtype import Order, Position
from fintorch.enum import OrderSide, PositionSide


def prepare_dataframe_for_mpf(df: pd.DataFrame) -> pd.DataFrame:
    pdf = df.copy()
    pdf['datetime'] = [datetime.fromtimestamp(ts) for ts in df.index.to_list()]
    pdf = pdf[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    pdf.set_index("datetime", inplace=True)

    return pdf


def draw_candlestick_plot(ax: plt.Axes, df: pd.DataFrame) -> None:
    pdf = prepare_dataframe_for_mpf(df=df)
    mpf.plot(pdf, type='candle', volume=False, ax=ax, style='binance')


def draw_order(ax: plt.Axes, df: pd.DataFrame, order: Order) -> None:
    df = df.reset_index()
    df.loc[df.timestamp == order.timestamp, "price"] = order.price
    color = 'g' if OrderSide.BUY == order.side else 'r'
    ax.axhline(y=df.close.mean(), xmin=xmin, xmax=len(df), color=color, linestyle='-')


def draw_position(ax: plt.Axes, df: pd.DataFrame, position: Position) -> None:
    df = df.reset_index()
    xmin = df[df.timestamp == position.entry_timestamp].index[0]
    color = 'g' if PositionSide.LONG == position.side else 'r'
    ax.axhline(y=df.close.mean(), xmin=xmin, xmax=len(df), color=color, linestyle='-')


def draw_rectangle(ax: plt.Axes, x1: int, y1: int, x2: int, y2: int, color: str = None, alpha: float = 0.25):
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    color = color if color else 'g' if y1 < y2 else 'r'
    area = Rectangle((x, y), width, height, color=color, alpha=alpha)
    ax.add_patch(area)


def draw_labels(ohlc_ax: plt.Axes, df: pd.DataFrame):
    label, start_index = df.label.iloc[0], 0
    for index in range(1, len(df) + 1):
        if len(df) == index or label != df.label.iloc[index]:
            minimum_price = df.low.iloc[start_index: index].min()
            maximum_price = df.high.iloc[start_index: index].max()
            y1 = minimum_price if 1 == label else maximum_price
            y2 = maximum_price if 1 == label else minimum_price
            draw_rectangle(ax=ohlc_ax, x1=start_index, y1=y1, x2=index, y2=y2)

            if index < len(df):
                label = df.label.iloc[index]
                start_index = index


def draw_predictions(ohlc_ax: plt.Axes, df: pd.DataFrame):
    label, start_index, y_values = df.label.iloc[0], 0, []
    for index in range(1, len(df) + 1):
        if len(df) == index or label != df.label.iloc[index]:
            minimum_price = df.low.iloc[start_index: index].min()
            maximum_price = df.high.iloc[start_index: index].max()

            for j in range(start_index, index):
                y_values.append(minimum_price if 1 == label else maximum_price)

            if index < len(df):
                label = df.label.iloc[index]
                start_index = index

    # create up and down data frames
    df = df.copy()
    df["y"] = y_values
    udf = df[1 == df.prediction]
    ddf = df[0 == df.prediction]

    # draw scatters
    ohlc_ax.scatter(x=udf.index.to_list(), y=udf.y, s=2, marker='o', c="g", label="Up Prediction")
    ohlc_ax.scatter(x=ddf.index.to_list(), y=ddf.y, s=2, marker='o', c="r", label="Down Prediction")
