from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.dataset.data import Data
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.fmsma import FMsmaLabelTransform


def main():
    symbols = ['BTC-USDT', 'ETH-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.HOUR1, TimeFrame.HOUR4]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = FMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[1])
    dataset = BsfDataset(samples_count=1000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)
    dataset.prepare(data=data)


if __name__ == '__main__':
    main()
