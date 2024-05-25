from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=args.time_frames)

    # create fold iterator
    first_timestamp = args.dataset.get_start_timestamp(dc=dc)
    last_timestamp = args.dataset.get_stop_timestamp(dc=dc)
    fold_iterator = args.cross_validation(first_timestamp=first_timestamp, last_timestamp=last_timestamp)
    fold = next(fold_iterator)

    # define data loader and load single batch
    with args.dataset:
        batch_iterator = args.data_loader(dataset=args.dataset, timestamps=fold.train_timestamps, shuffle=False)
        batch_x, batch_y = next(iter(batch_iterator))

        print("{:^32}{:^32}{:^32}{:^32}".format("Model", "Batch x", "Batch y", "Batch y hat"))
        print("{:^32}{:^32}{:^32}{:^32}".format("-" * 28, "-" * 28, "-" * 28, "-" * 28))
        for model in args.models:
            output = model(batch_x)
            print("{:^32}{:^32}{:^32}{:^32}"
                  .format(model.short_name, str(batch_x.shape), str(batch_y.shape), str(output.shape)))


if __name__ == '__main__':
    main()
