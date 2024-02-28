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
    with dataset:
        for index in range(0, len(timestamps), args.batch_size):
            start_time = time.perf_counter()
            batch_timestamps = timestamps[index:index + args.batch_size]
            batch_x, batch_y = dataset[batch_timestamps]
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000
            print("Loading batch takes:     {:.3f} ms".format(elapsed_time_ms))
            print("Batch x.shape:", batch_x.shape)
            print("Batch y.shape:", batch_y.shape)


if __name__ == '__main__':
    main()
