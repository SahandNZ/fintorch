import argparse
import json
import time

from examples.args import add_default_args_and_parse
from fintorch.data import load_data
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *


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

    # load candlestick data
    data = load_data(symbols=symbols, time_frames=time_frames)

    # preprocess single feature
    start_time = time.perf_counter()
    feature = dataset.preprocess(data=data, timestamp=dataset.timestamps[-1])
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print("Preprocessing takes: {:.3f} ms".format(elapsed_ms))
    print("Feature.shape:", feature.shape)


if __name__ == '__main__':
    main()
