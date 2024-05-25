import time

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=args.time_frames)

    # load feature and label
    start_time = time.perf_counter()
    with args.dataset:
        batch_timestamps = args.dataset.get_timestamps(dc=dc)[:args.batch_size]
        feature, label = args.dataset.load(timestamps=batch_timestamps)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print("Loading takes: {:.3f} ms".format(elapsed_ms))
    print("Feature.shape:", feature.shape)
    print("Label.shape:", label.shape)


if __name__ == '__main__':
    main()
