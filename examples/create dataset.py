from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_msma import ForwardMsmaLabelTransform


def main():
    exchange = 'binance'
    symbols = ['ETH-USDT']
    time_frames = [TimeFrame.HOUR1]

    # preparing data
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # create and save datasets
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = ForwardMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)
    dataset.prepare(data=data, samples_count=0, show_progress_bar=True)


if __name__ == '__main__':
    main()
