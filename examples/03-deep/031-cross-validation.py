from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.transform.feature import RollingMeanStdTrRocFeatureTransform
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.function import call_with_dict


def main():
    args = DefaultArgumentParser.parse()

    # define feature transform
    feature_transform = call_with_dict(RollingMeanStdTrRocFeatureTransform, args.transform_kwargs)

    # define sliding window cross validation
    cross_validation = CrossValidation(interval=args.interval)

    # load on board timestamp
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    first_valid_timestamp = feature_transform.get_first_valid_timestamp(dc=dc)

    # iterate over folds
    for fold in cross_validation(first_valid_timestamp=first_valid_timestamp):
        print("{:<32}: {}".format("Fold train start datetime", fold.train_start_datetime))
        print("{:<32}: {}".format("Fold validation start datetime", fold.val_start_datetime))
        print("{:<32}: {}".format("Fold test start datetime", fold.test_start_datetime))
        print("{:<32}: {}\n".format("Fold test stop datetime", fold.test_stop_datetime))


if __name__ == '__main__':
    main()
