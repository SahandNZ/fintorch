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
        print("{:<32}: {}".format("Fold train start datetime", fold.train_start_datetime))
        print("{:<32}: {}".format("Fold validation start datetime", fold.val_start_datetime))
        print("{:<32}: {}".format("Fold test start datetime", fold.test_start_datetime))
        print("{:<32}: {}\n".format("Fold test stop datetime", fold.test_stop_datetime))


if __name__ == '__main__':
    main()
