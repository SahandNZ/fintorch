import argparse
import json

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import SfDataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(sequence_length=args.dim_sequence)
    label_transform = UpDownLabelTransform()

    # define dataset
    dataset = SfDataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        interval=args.interval,
        symbol=symbols[0],
        time_frame=time_frames[0],
        sequence_length=args.dim_sequence,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    # define cross validation
    cross_validation = CrossValidation(train_percentage=80, val_percentage=10)

    for fold in cross_validation(dataset=dataset):
        print("=" * 32)
        print(fold.train_start_datetime, fold.validation_start_datetime, fold.test_start_datetime,
              fold.test_stop_datetime)
        data_loader = DataLoader()
        for batch_x, batch_y in data_loader(dataset=dataset, timestamps=fold.train_timestamps, batch_size=128):
            print("\t- ", batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
