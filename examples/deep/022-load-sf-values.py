import argparse
import json
import time

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
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

    start_time = time.perf_counter()
    x, y = dataset[dataset.timestamps]
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    print("Loading samples takes:     {:.3f} ms".format(elapsed_time_ms))
    print("Loading each sample takes: {:.3f} ms".format(elapsed_time_ms / len(dataset.timestamps)))


if __name__ == '__main__':
    main()
