import time
from datetime import datetime

import numpy as np

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=args.time_frames)

    start_time = time.time()
    with args.module:
        timestamps = args.module.get_timestamps(dc=dc, start_date=args.start_date, stop_date=args.stop_date)
        y_hat_dict = args.module.predict(dc=dc, timestamps=timestamps)
    elapsed_time_ms = (time.time() - start_time) * 1000

    print("Loading predictions takes: {:.3f} ms".format(elapsed_time_ms))
    print("Nan values count:", np.isnan(np.array(list(y_hat_dict.values()))).sum())
    print(len(timestamps), len(y_hat_dict))
    for ts, y_hat in y_hat_dict.items():
        if np.isnan(y_hat).max():
            print(datetime.fromtimestamp(ts))


if __name__ == '__main__':
    main()
