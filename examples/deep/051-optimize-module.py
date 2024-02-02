import argparse
import json

from rich.live import Live
from rich.panel import Panel
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import SfDataset
from fintorch.deep.model import FeedForward
from fintorch.deep.module import Module
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

    with Live(refresh_per_second=2) as live:
        for fold in module.optimize_and_store():
            live.update(Panel.fit(str(fold), title=f"[blue]{str(module)}"))


if __name__ == '__main__':
    main()
