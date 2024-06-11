from datetime import datetime

import numpy as np
import torch

from fintorch.deep.metrics import Metrics
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.timestamp import create_timestamps, to_timestamp


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(
        symbols=[args.module.symbol],
        time_frames=args.module.dataset.time_frames
    )

    # create y and y_hat values
    with args.module:
        for fold in args.module.folds_dict.values():
            print()
            print("#" * 32)
            print("#" * 32)
            print("#" * 32)
            print()
            print(fold.start_date, fold.stop_date)
            print(fold.best_val_epoch.model_state_dict)
                


if __name__ == '__main__':
    main()
