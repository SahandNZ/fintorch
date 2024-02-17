import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.cross_validation import CrossValidation
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import RollingMeanStdTrRocFeatureTransform
from fintorch.deep.transform.label import ForwardMiddleSmaLabelTransform
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # define sliding window cross validation
    cross_validation = CrossValidation(interval=args.interval)

    # load on board timestamp
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    symbol_info = dc.get_symbol_info(symbol=args.symbol)
    on_board_timestamp = symbol_info.on_board_timestamp

    # iterate over folds
    for fold in cross_validation(on_board_timestamp=on_board_timestamp):
        print("{:<32}: {}".format("Fold train start datetime", fold.train_start_datetime))
        print("{:<32}: {}".format("Fold validation start datetime", fold.val_start_datetime))
        print("{:<32}: {}".format("Fold test start datetime", fold.test_start_datetime))
        print("{:<32}: {}\n".format("Fold test stop datetime", fold.test_stop_datetime))


if __name__ == '__main__':
    main()
