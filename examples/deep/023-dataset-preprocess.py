import argparse
import itertools
import json
import os.path
import time

from rich.progress import Progress

from fintorch.data import load_data
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
from fintorch.deep.module import EncoderModule
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-batch-norm", action="store_false")
    parser.add_argument("--dim-latent", action="store", type=int, required=False, default=128)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=2)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    label_transform = UpDownLabelTransform()

    # create dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        sampling_time_frame=args.sampling_time_frame,
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    # load candlestick data
    data = load_data(symbols=symbols, time_frames=time_frames)

    # create single sample
    start_time = time.perf_counter()
    feature = dataset.preprocess(data=data, timestamp=dataset.timestamps[-1])
    print(time.perf_counter() - start_time)
    print(feature)


if __name__ == '__main__':
    main()
