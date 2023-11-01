from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.transform.label.classification.trend.forward_ichi import ForwardIchiLabelTransform


def main():
    symbol = 'BTC-USDT'
    time_frame = TimeFrame.HOUR1

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=[symbol], time_frames=[time_frame])
    data = Data(df_dict)

    # crop data
    start_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
    data = data.crop(start_timestamp=start_timestamp)

    # creating dataset
    label_transform = ForwardIchiLabelTransform(symbol=symbol, time_frame=time_frame)
    label_dataframe = label_transform.fit(data=data)
    current_open_timestamp = datetime.now().timestamp() // label_transform.time_frame * label_transform.time_frame
    timestamps = [label_dataframe.index.to_list()[-1], current_open_timestamp]
    labels = label_transform.transform(df=label_dataframe, timestamps=timestamps)

    # draw ohlcv plot
    fig = label_transform.draw_ohlcv_plot(data=data)
    plt.show()


if __name__ == '__main__':
    main()
