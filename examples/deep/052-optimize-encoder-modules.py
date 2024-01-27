import argparse
import itertools
import json
import os.path

from rich.progress import Progress

from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
from fintorch.deep.module import EncoderModule
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-batch-norm", action="store_false")
    parser.add_argument("--epochs", action="store", type=int, required=False, default=20)
    parser.add_argument("--batch-size", action="store", type=int, required=False, default=1024)
    parser.add_argument("--dim-latent", action="store", type=int, required=False, default=128)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=2)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length),
    ]

    # define label transform
    label_transform = UpDownLabelTransform()

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    for feature_transform, model_type in itertools.product(feature_transforms, model_types):
        print("=" * 64)
        print("{}{:^32}{}".format("=" * 16, feature_transform.short_name + " " + model_type.__name__, "=" * 16))
        print("=" * 64)

        with Progress(*RICH_PROGRESS_COLUMNS, transient=True) as progress:
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

            # define auto encoder
            encoder_module = EncoderModule(
                dataset=dataset,
                dim_latent=args.dim_latent,
                num_hidden_layers=args.num_hidden_layers,
                batch_norm=args.no_batch_norm,
                encoder=model_type
            )

            if not os.path.exists(encoder_module.path):
                encoder_module.optimize_and_store(epochs=args.epochs, batch_size=args.batch_size, progress=progress)


if __name__ == '__main__':
    main()
