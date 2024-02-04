import argparse
import json

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbol=args.symbol, time_frame=args.time_frame,
                                                            dim_sequence=args.dim_sequence)
    label_transform = ForwardMiddleSmaLabelTransform(symbol=args.symbol, time_frame=args.time_frame)

    # define dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        interval=args.interval,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Loading dataset samples", total=len(dataset.timestamps))
        for timestamp in dataset.timestamps:
            x, y = dataset[timestamp]
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
