import argparse
from typing import Type

from pyccx.data.local import load_dataframes_dict
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.transform.label.classification.trend.up_down import UpDownLabelTransform

exchange: str = None


def work(symbol: str, time_frame: int, feature_transform_cls: Type, progress: Progress = None):
    symbols = [symbol]
    time_frames = [time_frame]

    # load candlestick data
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # create dataset (create and store features for all timestamp)
    label_transform = UpDownLabelTransform(symbol=symbol, time_frame=time_frame)
    feature_transform = feature_transform_cls(symbols=symbols, time_frames=time_frames)
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)
    dataset.prepare(data=data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true", required=False)
    parser.add_argument("--show-progress-bar", action="store_true", required=False)
    parser.add_argument("--max-workers", action="store", type=int, required=False, default=64)
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="datasets.json")
    args = parser.parse_args()

    # set global variables
    global exchange
    exchange = args.exchange

    args.show_progress_bar


if __name__ == '__main__':
    main()
