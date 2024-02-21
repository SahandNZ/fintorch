import argparse
import itertools
import json

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.model import MODEL_TYPES
from fintorch.deep.module import create_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # get symbols and time_frames fom config_dict
    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define model params
    model_kwargs = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 2,
        "num_hidden_layers": args.num_hidden_layers,
        "batch_norm": args.no_batch_norm,
        "dropout": args.dropout,
        "activation_fn": nn.Softmax(dim=-1)
    }

    # define modules params
    items = list(itertools.product(symbols, time_frames, FEATURE_TRANSFORM_TYPES, LABEL_TRANSFORM_TYPES, MODEL_TYPES))

    # optimize modules with rich panel
    overall_progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task = overall_progress.add_task(description="overall jobs", total=len(items))
    progress_panel = Panel.fit(overall_progress, title="Overall progress")

    with Live(refresh_per_second=2) as live:
        for symbol, time_frame, ft_type, lt_type, model_type in items:
            # create module
            transform_kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
            module = create_module(
                feature_transform_type=ft_type,
                label_transform_type=lt_type,
                transform_kwargs=transform_kwargs,
                model_type=model_type,
                model_kwargs=model_kwargs,
                interval=args.interval,
            )

            # load data collection
            dc = ONLINE_EXCHANGE.future.data.get_data_collection(
                symbols=[module.symbol],
                time_frames=[module.time_frame]
            )

            # optimize folds
            for status in module.optimize(dc=dc):
                live.update(Group(Panel.fit(str(status), title=str(module)), progress_panel))

            overall_progress.update(overall_task, advance=1)


if __name__ == '__main__':
    main()
