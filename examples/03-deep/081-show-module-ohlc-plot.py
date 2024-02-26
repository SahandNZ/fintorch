import argparse
import copy
import itertools
import json

import numpy as np
import torch
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.metrics import Metrics
from fintorch.deep.model import FeedForward, Hybrid, MODEL_TYPES
from fintorch.deep.module import create_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature and label transforms
    transform_kwargs = {"symbol": args.symbol, "time_frame": args.time_frame, "dim_sequence": args.dim_sequence}
    feature_transform_type = RollingMeanStdTrRocFeatureTransform
    label_transform_type = ForwardMiddleSmaLabelTransform

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
    module = create_module(
        feature_transform_type=feature_transform_type,
        label_transform_type=label_transform_type,
        transform_kwargs=transform_kwargs,
        model_type=model_type,
        model_kwargs=model_kwargs,
        interval=args.interval,
    )

    # show ohlc plot
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])

    with module:
        module.show_ohlc_plot(dc=dc, start_date=args.start_date, stop_date=args.stop_date)


if __name__ == '__main__':
    main()
