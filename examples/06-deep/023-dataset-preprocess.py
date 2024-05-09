import time

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # preprocess single feature
    start_time = time.perf_counter()
    with args.dataset:
        timestamps = args.feature_transform.get_timestamps(dc=dc)
        feature = args.dataset.preprocess(dc=dc, timestamps=timestamps[-1024:])
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print("Preprocessing takes: {:.3f} ms".format(elapsed_ms))
    print("Feature.shape:", feature.shape)


if __name__ == '__main__':
    main()
