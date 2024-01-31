import argparse
import json

import torch

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import SfDataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
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
    label_transform = ForwardMiddleSmaLabelTransform()

    # create dataset
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

    # model dims
    model_dims = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 16,
    }

    # define cross validation and generate single fold
    cross_validation = CrossValidation(train_percentage=80, val_percentage=10)
    fold = next(iter(cross_validation(dataset=dataset)))

    # define data loader and load single batch
    data_loader = DataLoader()
    batch_x, _ = next(iter(data_loader(dataset=dataset, timestamps=fold.train_timestamps, batch_size=args.batch_size)))

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    print("{:^32}{:^32}{:^32}".format("Model", "Batch x", "Batch y hat"))
    print("{:^32}{:^32}{:^32}".format("-" * 28, "-" * 28, "-" * 28))
    for model_type in model_types:
        model = model_type(**model_dims)
        output = model(batch_x)
        print("{:^32}{:^32}{:^32}".format(model.name, str(batch_x.shape), str(output.shape)))


if __name__ == '__main__':
    main()
