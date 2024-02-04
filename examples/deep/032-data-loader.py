import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


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

    # define cross validation
    cross_validation = CrossValidation(train_percentage=80, val_percentage=10)

    for fold in cross_validation(dataset=dataset):
        print("=" * 32)
        print(fold.train_start_datetime, fold.val_start_datetime, fold.test_start_datetime, fold.test_stop_datetime)
        data_loader = DataLoader(batch_size=1024)
        for batch_x, batch_y in data_loader(dataset=dataset, timestamps=fold.train_timestamps):
            print("\t- ", batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
