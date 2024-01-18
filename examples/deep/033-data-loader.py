import argparse
import json

from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", action="store", type=int, required=False, default=256)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length)
    label_transform = UpDownLabelTransform()

    # create dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        sampling_time_frame=args.sampling_time_frame,
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    cross_validation = CrossValidation(train_percentage=80, dev_percentage=10)

    for fold in cross_validation(dataset=dataset):
        print("=" * 32)
        print(fold.train_start_datetime, fold.validation_start_datetime, fold.test_start_datetime,
              fold.test_stop_datetime)
        data_loader = DataLoader()
        for batch_x, batch_y in data_loader(dataset=dataset, timestamps=fold.train_timestamps, batch_size=128):
            print("\t- ", batch_x.shape, batch_y.shape)


if __name__ == '__main__':
    main()
