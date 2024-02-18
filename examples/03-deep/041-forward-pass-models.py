import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import MODEL_TYPES
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define feature and label transforms
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbol=args.symbol, time_frame=args.time_frame,
                                                            dim_sequence=args.dim_sequence)
    label_transform = ForwardMiddleSmaLabelTransform(symbol=args.symbol, time_frame=args.time_frame)

    # define dataset
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

    # model dims
    model_kwargs = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 16,
    }

    # load data collection and create timestamps
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    start_timestamp = feature_transform.get_start_timestamp(dc=dc)

    # define cross validation and generate single fold
    cross_validation = CrossValidation(interval=args.interval)
    fold = next(iter(cross_validation(start_timestamp=start_timestamp)))

    # define exchange loader and load single batch
    data_loader = DataLoader(batch_size=1024)
    batch_x, _ = next(iter(data_loader(dataset=dataset, timestamps=fold.train_timestamps, shuffle=False)))

    print("{:^32}{:^32}{:^32}".format("Model", "Batch x", "Batch y hat"))
    print("{:^32}{:^32}{:^32}".format("-" * 28, "-" * 28, "-" * 28))
    for model_type in MODEL_TYPES:
        model = model_type(**model_kwargs)
        output = model(batch_x)
        print("{:^32}{:^32}{:^32}".format(model.name, str(batch_x.shape), str(output.shape)))


if __name__ == '__main__':
    main()
