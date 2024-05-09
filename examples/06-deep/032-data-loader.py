from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # create fold iterator
    first_timestamp = args.dataset.feature_transform.get_start_timestamp(dc=dc)
    last_timestamp = args.dataset.label_transform.get_stop_timestamp(dc=dc)
    fold_iterator = args.cross_validation(first_timestamp=first_timestamp, last_timestamp=last_timestamp)

    # iterate over folds
    for fold in fold_iterator:
        print("=" * 32)
        print(fold.train_start_datetime, fold.val_start_datetime, fold.test_start_datetime, fold.test_stop_datetime)
        with args.dataset:
            batch_iterator = args.data_loader(dataset=args.dataset, timestamps=fold.train_timestamps, shuffle=False)
            for batch_x, batch_y in batch_iterator:
                print("\t- ", batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
