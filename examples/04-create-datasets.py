import argparse
import json

from pyccx.data import load_dataframes_dict
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.transform.feature import *
from fintorch.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=2 ** 8)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/datasets.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    ]

    label_transforms = [
        ForwardBackwardMinimumLabelTransform(),
        ForwardIchimokuLabelTransform(),
        ForwardMiddleSmaLabelTransform(),
        ForwardRocLabelTransform(),
        NextFractalLabelTransform(),
        UpDownLabelTransform()
    ]

    # create dataset
    dataset = Dataset(
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        sampling_time_frame=args.sampling_time_frame,
        feature_transforms=feature_transforms,
        label_transforms=label_transforms
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        # load candlestick data
        df_dict = load_dataframes_dict(args.exchange, symbols, time_frames, progress=progress)
        data = Data(df_dict)

        # creating samples
        dataset.prepare(data=data, start_date="2021-01-01", progress=progress)


if __name__ == '__main__':
    main()
