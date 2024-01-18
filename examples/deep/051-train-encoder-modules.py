import argparse
import json

from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, Hybrid
from fintorch.deep.module import EncoderModule
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", action="store", type=int, required=False, default=64)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--window-length", action="store", type=int, required=False, default=10000)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

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

    # dataset dims
    dim_symbol = len(symbols)
    dim_time_frame = len(time_frames)
    dim_sequence = args.sequence_length
    dim_feature = 4

    # define auto encoder
    encoder_module = EncoderModule(dim_symbol=dim_symbol, dim_time_frame=dim_time_frame, dim_sequence=dim_sequence,
                                   dim_feature=dim_feature, dim_latent=16, model=Hybrid)

    encoder_module.optimize(dataset=dataset)


if __name__ == '__main__':
    main()
