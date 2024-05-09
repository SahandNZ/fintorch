import time

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # load batch
    start_time = time.perf_counter()
    with args.dataset:
        timestamps = args.feature_transform.get_timestamps(dc=dc)
        x, y = args.dataset[timestamps]
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    print("Loading batch takes:     {:.3f} ms".format(elapsed_time_ms))
    print(x.shape, y.shape)


if __name__ == '__main__':
    main()
