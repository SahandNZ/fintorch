from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.bfm import BfmLabelTransform


def main():
    symbols = ['BTC-USDT', 'ETH-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.HOUR1, TimeFrame.HOUR4]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = BfmLabelTransform(symbol=symbols[0], time_frame=time_frames[1])
    dataset = BsfDataset(samples_count=1000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)
    dataset.prepare(data=data)
    print(dataset.x.shape)
    print(dataset.x.shape)
    print()

    # preprocess example
    timestamps = dataset.df.index.to_list()[-10:]
    x = dataset.preprocess(data=data, timestamps=timestamps, show_progress_bar=True)
    print(x.shape)


if __name__ == '__main__':
    main()
