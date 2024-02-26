import argparse

import numpy as np
import torch
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.metrics import Metrics
from fintorch.deep.model import FeedForward
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

    # calculate metrics
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    # create y and y_hat values
    with module:
        sf_generator = module.dataset.label_transform.transform_sf(dc=dc, timestamps=timestamps)
        y_array = np.concatenate([sf for sf in sf_generator])
        y_hat_dict = module.predict(dc=dc, timestamps=timestamps)

        # convert y and y_hat values to torch.Tensor
        y = torch.from_numpy(y_array)
        y_hat = torch.from_numpy(np.array(list(y_hat_dict.values())))

    # calculate metrics
    metrics = Metrics(criterion=module.trainer.criterion, y=y, y_hat=y_hat)
    print(round(metrics.objective, 4))
    print(metrics.accuracy)
    print(metrics.precision(label=0))
    print(metrics.precision(label=1))


if __name__ == '__main__':
    main()
