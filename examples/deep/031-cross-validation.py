import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation, SlidingWindowCrossValidation
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import RollingMeanStdTrRocFeatureTransform
from fintorch.deep.transform.label import ForwardMiddleSmaLabelTransform


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbol=args.symbol, time_frame=args.time_frame,
                                                            dim_sequence=args.dim_sequence)
    label_transform = ForwardMiddleSmaLabelTransform(symbol=args.symbol, time_frame=args.time_frame)

    # define dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        interval=args.interval,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    # define sliding window cross validation
    cross_validation = CrossValidation(train_percentage=80, val_percentage=10)
    cross_validation = SlidingWindowCrossValidation(train_percentage=50, val_percentage=20, window_length=50000)

    for fold in cross_validation(dataset=dataset):
        print("{:<32}: {}".format("Fold index", fold.index))
        print("{:<32}: {}".format("Fold train start datetime", fold.train_start_datetime))
        print("{:<32}: {}".format("Fold validation start datetime", fold.val_start_datetime))
        print("{:<32}: {}".format("Fold test start datetime", fold.test_start_datetime))
        print("{:<32}: {}\n".format("Fold test stop datetime", fold.test_stop_datetime))


if __name__ == '__main__':
    main()
