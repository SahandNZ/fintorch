from datetime import datetime

from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.dataset.data import Data
from fintorch.transform.label.classification.trend.fmsma import FMsmaLabelTransform


def main():
    symbols = ['BTC-USDT', 'ETH-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.HOUR1, TimeFrame.HOUR4]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # creating dataset
    label_transform = FMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[1])
    label_dataframe = label_transform.fit(data=data)
    current_open_timestamp = datetime.now().timestamp() // label_transform.time_frame * label_transform.time_frame
    timestamps = [label_dataframe.index.to_list()[-1], current_open_timestamp]
    labels = label_transform.transform(df=label_dataframe, timestamps=timestamps)

    print(labels)


if __name__ == '__main__':
    main()
