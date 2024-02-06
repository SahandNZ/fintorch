import argparse

from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward
from fintorch.deep.module import Module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.utils.timestamp import create_timestamps


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

    # define model
    model = FeedForward(
        dim_sequence=args.dim_sequence,
        dim_feature=4,
        dim_output=2,
        num_hidden_layers=args.num_hidden_layers,
        batch_norm=args.no_batch_norm,
        dropout=args.dropout,
        activation_fn=nn.Softmax(dim=-1)
    )

    # define module
    module = Module(
        dataset=dataset,
        model=model
    )

    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    y_hat = module.predict(timestamps=timestamps)


if __name__ == '__main__':
    main()
