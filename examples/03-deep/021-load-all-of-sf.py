import time

from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.function import call_with_dict


def main():
    args = DefaultArgumentParser.parse()

    # define feature and label transforms
    feature_transform = call_with_dict(RollingMeanStdTrRocFeatureTransform, args.transform_kwargs)
    label_transform = call_with_dict(ForwardMiddleSmaLabelTransform, args.transform_kwargs)

    # define dataset
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

    # load data collection and create timestamps
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    timestamps = feature_transform.get_valid_timestamps(dc=dc)

    # load batch
    start_time = time.perf_counter()
    with dataset:
        _, _ = dataset[timestamps]
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    print("Loading batch takes:     {:.3f} ms".format(elapsed_time_ms))


if __name__ == '__main__':
    main()
