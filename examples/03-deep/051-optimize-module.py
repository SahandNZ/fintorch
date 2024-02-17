import argparse

from rich.live import Live
from rich.panel import Panel
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward
from fintorch.deep.module import Module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbol=args.symbol, time_frame=args.time_frame,
                                                            dim_sequence=args.dim_sequence)
    label_transform = ForwardMiddleSmaLabelTransform(symbol=args.symbol, time_frame=args.time_frame)

    # define dataset
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

    # define model type and kwargs
    model_type = FeedForward
    model_kwargs = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 2,
        "num_hidden_layers": args.num_hidden_layers,
        "batch_norm": args.no_batch_norm,
        "dropout": args.dropout,
        "activation_fn": nn.Softmax(dim=-1)
    }

    # define module
    module = Module(dataset=dataset, model_type=model_type, model_kwargs=model_kwargs)

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    with Live(refresh_per_second=2) as live:
        for fold in module.optimize(dc=dc):
            live.update(Panel.fit(str(fold), title=f"[blue]{str(module)}"))


if __name__ == '__main__':
    main()
