import argparse
import json
import time

import torch

from examples.args import add_default_args_and_parse
from fintorch.data import load_data
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # load symbols and time_frames from config file
    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length)
    label_transform = UpDownLabelTransform()

    # create dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        interval=args.interval,
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    # define auto encoder
    encoder_module = EncoderModule(
        dataset=dataset,
        dim_latent=args.dim_latent,
        num_hidden_layers=args.num_hidden_layers,
        batch_norm=args.no_batch_norm,
        model_type=FeedForward
    )

    # prepare inputs (load candlestick data and preprocess x)
    data = load_data(symbols=symbols, time_frames=time_frames)
    x = dataset.preprocess(data=data, timestamps=dataset.timestamps)

    # encoder module forward pass
    start_time = time.perf_counter()
    latent_x = encoder_module.predict(x=x)
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    mstd = torch.mean(torch.std(latent_x, dim=-1), dim=0)

    print("Encoding latent x takes: {:.3f} ms".format(elapsed_time_ms))
    print("Latent_x.shape: {}".format(latent_x.shape))
    print("MSTD latent_x: {:.2f}".format(mstd))


if __name__ == '__main__':
    main()
