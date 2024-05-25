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
        stop_timestamp = args.module.dataset.label_transform.get_stop_timestamp(dc=dc)
        stop_date = datetime.fromtimestamp(min(stop_timestamp, to_timestamp(args.stop_date)))
        timestamps = create_timestamps(start_date=args.start_date, stop_date=stop_date,
                                       time_frame=args.module.time_frame)

        sf_generator = args.module.dataset.label_transform.transform_sf(dc=dc, timestamps=timestamps)
        y_array = np.concatenate([sf for sf in sf_generator])
        y_hat_dict = args.module.predict(dc=dc, timestamps=timestamps)

        # convert y and y_hat values to torch.Tensor
        y = torch.from_numpy(y_array)
        y_hat = torch.from_numpy(np.array(list(y_hat_dict.values())))

    # calculate metrics
    metrics = Metrics(criterion=args.module.trainer.criterion, y=y, y_hat=y_hat)
    print(metrics)


if __name__ == '__main__':
    main()
