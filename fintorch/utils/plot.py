from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ..dtype import Order, Position
from ..enum import OrderType, PositionSide


def draw_candlestick_plot(ax: plt.Axes, df: pd.DataFrame) -> None:
    # prepare date frame for mpf
    pdf = df.copy()
    pdf["datetime"] = [datetime.fromtimestamp(ts) for ts in df.index.to_list()]
    pdf = pdf[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    pdf.set_index("datetime", inplace=True)

    # draw candlestick plot with mpf
    mpf.plot(pdf, type='candle', volume=False, ax=ax, style='binance')

    # add title and labels
    ax.set_ylabel("Price")
    ax.set_xlabel("Time")
    ax.grid()


def draw_rectangle(ax: plt.Axes, x1: int, y1: float, x2: int, y2: float, color: str = None, alpha: float = 0.25):
    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    color = color or ('g' if y1 < y2 else 'r')
    area = Rectangle((x, y), width, height, color=color, alpha=alpha)
    ax.add_patch(area)


def draw_labels(ohlc_ax: plt.Axes, df: pd.DataFrame):
    label, start_index = df.label.iloc[0], 0
    for index in range(1, len(df) + 1):
        if len(df) == index or label != df.label.iloc[index]:
            if not np.isnan(label):
                minimum_price = df.low.iloc[start_index: index].min()
                maximum_price = df.high.iloc[start_index: index].max()
                y1 = minimum_price if 1 == label else maximum_price
                y2 = maximum_price if 1 == label else minimum_price
                draw_rectangle(ax=ohlc_ax, x1=start_index, y1=y1, x2=index - 1, y2=y2)

            if index < len(df):
                label = df.label.iloc[index]
                start_index = index


def draw_predictions_based_on_labels(ohlc_ax: plt.Axes, df: pd.DataFrame):
    label, start_index, y, y_values = df.label.iloc[0], 0, 0, []
    for index in range(1, len(df) + 1):
        if len(df) == index or label != df.label.iloc[index]:
            if not np.isnan(label):
                y = df.low.iloc[start_index: index].min()

            for j in range(start_index, index):
                y_values.append(y)

            if index < len(df):
                label = df.label.iloc[index]
                start_index = index

    # create up and down data frames
    df = df.copy()
    df["y"] = y_values
    udf = df[1 == df.prediction]
    ddf = df[0 == df.prediction]
    ndf = df[np.isnan(df.prediction)]

    # draw scatters
    size = 2 ** 11 // len(df)
    ohlc_ax.scatter(x=udf.index.to_list(), y=udf.y, s=size, marker='o', c="g", label="Up Prediction")
    ohlc_ax.scatter(x=ddf.index.to_list(), y=ddf.y, s=size, marker='o', c="r", label="Down Prediction")
    ohlc_ax.scatter(x=ndf.index.to_list(), y=ndf.y, s=size, marker='o', c="gray", label="Nan Prediction")


def draw_position_and_orders(ohlc_ax: plt.Axes, df: pd.DataFrame, position: Position, exit_orders: List[Order]) -> None:
    FONTSIZE = 12
    timestamp_to_index = {timestamp: i for i, timestamp in enumerate(df.index)}

    # draw position properties
    color = "g" if PositionSide.LONG == position.side else "r"
    position_y = position.entry_price
    position_xmin = timestamp_to_index[position.entry_timestamp]
    position_xmax = timestamp_to_index.get(position.exit_timestamp, len(df) - 1)
    text = f" {position_y}"

    ohlc_ax.hlines(
        y=position_y,
        xmin=position_xmin,
        xmax=position_xmax,
        color=color,
        linestyle='-',
        linewidth=2,
        label=f"{str(position.side).title()} Entry price"
    )

    ohlc_ax.text(
        x=position_xmax,
        y=position_y,
        s=text,
        color=color,
        fontsize=FONTSIZE,
        horizontalalignment='left',
        verticalalignment='center'
    )

    # draw exit orders properties
    for order in exit_orders:
        # Take profit orders
        if OrderType.LIMIT == order.type:
            order_y = order.price
            order_xmin = timestamp_to_index[order.timestamp]
            order_xmax = timestamp_to_index[order.close_timestamp]
            color = "g"
            percentage = (order_y / position_y - 1) * int(position.side) * 100
            text = "{} ({:.2f}%) ".format(position_y, percentage)

            ohlc_ax.hlines(
                y=order_y,
                xmin=order_xmin,
                xmax=order_xmax,
                color=color,
                linestyle='-',
                linewidth=2,
                label=order.comment
            )

            ohlc_ax.text(
                x=order_xmin,
                y=order_y,
                s=text,
                color=color,
                fontsize=FONTSIZE,
                horizontalalignment='right',
                verticalalignment='center'
            )

            draw_rectangle(
                ax=ohlc_ax,
                x1=order_xmin,
                y1=position_y,
                x2=order_xmax,
                y2=order_y,
                color=color
            )

        # Stop loss order
        elif OrderType.STOP_MARKET == order.type:
            order_y = order.stop_price
            order_xmin = timestamp_to_index[order.timestamp]
            order_xmax = timestamp_to_index[order.close_timestamp]
            color = "r"
            percentage = (order_y / position_y - 1) * int(position.side) * 100
            text = "{} ({:.2f}%) ".format(position_y, percentage)

            ohlc_ax.hlines(
                y=order_y,
                xmin=order_xmin,
                xmax=order_xmax,
                color=color,
                linestyle='-',
                linewidth=2,
                label=order.comment
            )

            ohlc_ax.text(
                x=order_xmin,
                y=order_y,
                s=text,
                color=color,
                fontsize=FONTSIZE,
                horizontalalignment='right',
                verticalalignment='center'
            )

            draw_rectangle(
                ax=ohlc_ax,
                x1=order_xmin,
                y1=position_y,
                x2=order_xmax,
                y2=order_y,
                color=color
            )

        # Exit order
        else:
            order_y = order.filled_price
            order_xmin = timestamp_to_index[order.close_timestamp]
            color = "g" if 0 < position.profit_percentage else "r"
            percentage = (order_y / position_y - 1) * int(position.side) * 100
            text = "{} ({:.2f}%) ".format(position_y, percentage)

            ohlc_ax.scatter(
                x=order_xmin,
                y=order_y,
                s=FONTSIZE,
                marker='x',
                c=color,
                label=order.comment
            )

            ohlc_ax.text(
                x=order_xmin,
                y=order_y,
                s=text,
                color=color,
                fontsize=FONTSIZE,
                horizontalalignment='right',
                verticalalignment='center'
            )

            draw_rectangle(
                ax=ohlc_ax,
                x1=position_xmin,
                y1=position_y,
                x2=position_xmax,
                y2=order_y,
                color=color
            )

    ohlc_ax.legend()
