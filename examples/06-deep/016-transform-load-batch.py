import time

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    timestamps = args.feature_transform.get_timestamps(dc=dc)

    # load batch
    with args.dataset:
        for index in range(0, len(timestamps), args.batch_size):
            start_time = time.perf_counter()
            batch_timestamps = timestamps[index:index + args.batch_size]
            batch_x, batch_y = args.dataset[batch_timestamps]
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000
            print("Loading batch takes:     {:.3f} ms".format(elapsed_time_ms))
            print("Batch x.shape:", batch_x.shape)
            print("Batch y.shape:", batch_y.shape)
            print()


if __name__ == '__main__':
    main()
