import argparse
import json

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import SfDataset
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(sequence_length=args.dim_sequence)
    label_transform = ForwardMiddleSmaLabelTransform()

    # define dataset
    dataset = SfDataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        interval=args.interval,
        symbol=symbols[0],
        time_frame=time_frames[0],
        sequence_length=args.dim_sequence,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Loading dataset samples", total=len(dataset.timestamps))
        for timestamp in dataset.timestamps:
            sample = dataset[timestamp]
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
