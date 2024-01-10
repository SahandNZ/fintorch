import argparse
import json
import math
from datetime import datetime

from rich.progress import Progress

from fintorch.dataset.dataset import Dataset
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.transform.feature import *
from fintorch.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-count", action="store", type=int, required=False, default=10000)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/config.json")
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

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    start_timestamp = math.ceil(start_date.timestamp() / args.sampling_time_frame) * args.sampling_time_frame
    stop_timestamp = start_timestamp + args.samples_count * args.sampling_time_frame
    timestamps = list(range(start_timestamp, stop_timestamp, args.sampling_time_frame))

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Loading dataset samples", total=len(timestamps))
        for timestamp in timestamps:
            sample = dataset[timestamp]
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
