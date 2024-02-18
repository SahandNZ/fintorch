import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.transform.feature import RollingMeanStdTrRocFeatureTransform
from fintorch.exchange import ONLINE_EXCHANGE


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature transform
    feature_transform = RollingMeanStdTrRocFeatureTransform(
        symbol=args.symbol,
        time_frame=args.time_frame,
        dim_sequence=args.dim_sequence
    )

    # define sliding window cross validation
    cross_validation = CrossValidation(interval=args.interval)

    # load on board timestamp
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    start_timestamp = feature_transform.get_start_timestamp(dc=dc)

    # iterate over folds
    for fold in cross_validation(start_timestamp=start_timestamp):
        print("{:<32}: {}".format("Fold train start datetime", fold.train_start_datetime))
        print("{:<32}: {}".format("Fold validation start datetime", fold.val_start_datetime))
        print("{:<32}: {}".format("Fold test start datetime", fold.test_start_datetime))
        print("{:<32}: {}\n".format("Fold test stop datetime", fold.test_stop_datetime))


if __name__ == '__main__':
    main()
