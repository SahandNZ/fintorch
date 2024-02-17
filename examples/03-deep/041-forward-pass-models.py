import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.timestamp import create_timestamps


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
    model_dims = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 16,
    }

    # load on board timestamp
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    symbol_info = dc.get_symbol_info(symbol=args.symbol)
    on_board_timestamp = symbol_info.on_board_timestamp

    # define cross validation and generate single fold
    cross_validation = CrossValidation(interval=args.interval)
    fold = next(iter(cross_validation(on_board_timestamp=on_board_timestamp)))

    # define exchange loader and load single batch
    data_loader = DataLoader(batch_size=1024)
    batch_x, _ = next(iter(data_loader(dataset=dataset, timestamps=fold.train_timestamps, shuffle=False)))

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    print("{:^32}{:^32}{:^32}".format("Model", "Batch x", "Batch y hat"))
    print("{:^32}{:^32}{:^32}".format("-" * 28, "-" * 28, "-" * 28))
    for model_type in model_types:
        model = model_type(**model_dims)
        output = model(batch_x)
        print("{:^32}{:^32}{:^32}".format(model.name, str(batch_x.shape), str(output.shape)))


if __name__ == '__main__':
    main()
