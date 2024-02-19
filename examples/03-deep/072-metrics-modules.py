import argparse
import itertools
import json
import time

from rich.progress import Progress
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.model import FeedForward, MODEL_TYPES
from fintorch.deep.module import create_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
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
        timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

        # create y and y_hat values
        y = [sf for sf in module.dataset.label_transform.transform_sf(dc=dc, timestamps=timestamps)]
        y_hat_dict = module.predict(dc=dc, timestamps=timestamps)
        y_hat = list(y_hat_dict.values())

        print(len(y), len(y))


if __name__ == '__main__':
    main()
