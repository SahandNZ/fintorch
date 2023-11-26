from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.feature.stft_tr_roc import StftTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_msma import ForwardMsmaLabelTransform


def main():
    exchange = 'binance'
    symbols = ['BTC-USDT']
    time_frames = [TimeFrame.MIN5]

    # preparing data
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # create and save datasets
    feature_transforms = [RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames),
                          StftTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)]
    label_transforms = [ForwardMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])]
    Dataset.create_and_save_datasets(dataset=BsfDataset, feature_transforms=feature_transforms,
                                     label_transforms=label_transforms, data=data, root="./data/datasets")


if __name__ == '__main__':
    main()
