import argparse
import json

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward
from fintorch.deep.module import EncoderModule
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform = StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    label_transform = UpDownLabelTransform()

    # create dataset
    dataset = Dataset(
        start_date=args.start_date,
        stop_date=args.stop_date,
        sampling_time_frame=args.sampling_time_frame,
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        feature_transform=feature_transform,
        label_transform=label_transform
    )

    # define encoder module
    encoder_module = EncoderModule(
        dataset=dataset,
        dim_latent=args.dim_latent,
        num_hidden_layers=args.num_hidden_layers,
        batch_norm=args.no_batch_norm,
        model_type=FeedForward
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        encoder_module.optimize_and_store(epochs=args.epochs, batch_size=args.batch_size, progress=progress)


if __name__ == '__main__':
    main()
