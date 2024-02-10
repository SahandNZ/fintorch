import argparse
import itertools
import json

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
from fintorch.deep.module import Module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.function import call_with_dict
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # get symbols and time_frames fom config_dict
    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform_types = [
        RollingMeanStdTrRocFeatureTransform,
        StftTrRocFeatureTransform
    ]

    label_transform_types = [
        ForwardBackwardMinimumLabelTransform,
        ForwardIchimokuLabelTransform,
        ForwardMiddleSmaLabelTransform,
        ForwardRocLabelTransform,
        NextFractalLabelTransform,
        UpDownLabelTransform
    ]

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    # define model params
    model_params = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 2,
        "num_hidden_layers": args.num_hidden_layers,
        "batch_norm": args.no_batch_norm,
        "dropout": args.dropout,
        "activation_fn": nn.Softmax(dim=-1)
    }

    # define timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    # define modules params
    items = list(itertools.product(symbols, time_frames, feature_transform_types, label_transform_types, model_types))

    # optimize modules with rich panel
    overall_progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task = overall_progress.add_task(description="overall jobs", total=len(items))
    progress_panel = Panel.fit(overall_progress, title="Overall progress")

    with Live(refresh_per_second=2) as live:
        for symbol, time_frame, ft_type, lt_type, model_type in items:
            # define feature and label transforms
            transform_param = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
            feature_transform = call_with_dict(ft_type, transform_param)
            label_transform = call_with_dict(lt_type, transform_param)

            # define dataset
            dataset = Dataset(
                start_date=args.start_date,
                stop_date=args.stop_date,
                interval=args.interval,
                feature_transform=feature_transform,
                label_transform=label_transform
            )

            # define model
            model = call_with_dict(model_type, model_params)

            # define module
            module = Module(dataset=dataset, model=model)

            # optimize module
            for fold in module.optimize_and_store(timestamps=timestamps):
                live.update(Group(Panel.fit(str(fold), title=str(module)), progress_panel))

            overall_progress.update(overall_task, advance=1)


if __name__ == '__main__':
    main()
