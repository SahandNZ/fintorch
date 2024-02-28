from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.data_loader import DataLoader
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import MODEL_TYPES
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.function import call_with_dict


def main():
    args = DefaultArgumentParser.parse()

    # define feature and label transforms and dataset
    feature_transform = call_with_dict(RollingMeanStdTrRocFeatureTransform, args.transform_kwargs)
    label_transform = call_with_dict(ForwardMiddleSmaLabelTransform, args.transform_kwargs)
    dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

    # load data collection and create timestamps
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    first_valid_timestamp = feature_transform.get_first_valid_timestamp(dc=dc)

    # define cross validation and generate single fold
    cross_validation = CrossValidation(interval=args.interval)
    fold = next(iter(cross_validation(first_valid_timestamp=first_valid_timestamp)))

    # define exchange loader and load single batch
    data_loader = DataLoader(batch_size=args.batch_size)
    with dataset:
        batch_x, _ = next(iter(data_loader(dataset=dataset, timestamps=fold.train_timestamps, shuffle=False)))

    print("{:^32}{:^32}{:^32}".format("Model", "Batch x", "Batch y hat"))
    print("{:^32}{:^32}{:^32}".format("-" * 28, "-" * 28, "-" * 28))
    for model_type in MODEL_TYPES:
        model = call_with_dict(model_type, args.model_kwargs)
        output = model(batch_x)
        print("{:^32}{:^32}{:^32}".format(model.name, str(batch_x.shape), str(output.shape)))


if __name__ == '__main__':
    main()
