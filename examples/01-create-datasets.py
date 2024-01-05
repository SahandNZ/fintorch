import argparse
import json

from pyccx.data import load_dataframes_dict
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.transform.feature import FEATURE_TRANSFORMS
from fintorch.transform.label import LABEL_TRANSFORMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=2 ** 8)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=3600)
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/datasets.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # create dataset
    dataset = Dataset(
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        sampling_time_frame=args.sampling_time_frame,
        feature_transforms=FEATURE_TRANSFORMS,
        label_transforms=LABEL_TRANSFORMS
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        # load candlestick data
        df_dict = load_dataframes_dict(args.exchange, symbols, time_frames, update=False, progress=progress)
        data = Data(df_dict)

        # creating samples
        dataset.prepare(data=data, progress=progress)


if __name__ == '__main__':
    main()
