import argparse

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.exchange import ONLINE_EXCHANGE
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
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

    # load data collection and create timestamps
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    timestamps = feature_transform.get_valid_timestamps(dc=dc)

    # prepare dataset
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Loading dataset samples", total=len(timestamps))
        dataset.prepare(dc=dc, timestamps=timestamps)
        for timestamp in timestamps:
            x, y = dataset[timestamp]
            print(x, y)
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
