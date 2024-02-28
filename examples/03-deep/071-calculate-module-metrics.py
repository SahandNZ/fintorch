import numpy as np
import torch

from fintorch.deep.metrics import Metrics
from fintorch.deep.model import FeedForward
from fintorch.deep.module import create_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    # define module
    module = create_module(
        feature_transform_type=RollingMeanStdTrRocFeatureTransform,
        label_transform_type=ForwardMiddleSmaLabelTransform,
        model_type=FeedForward,
        transform_kwargs=args.transform_kwargs,
        model_kwargs=args.model_kwargs,
        interval=args.interval,
    )

    # calculate metrics
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])
    valid_test_timestamps = module.get_vaid_test_timestamps(dc=dc)

    # create y and y_hat values
    with module:
        sf_generator = module.dataset.label_transform.transform_sf(dc=dc, timestamps=valid_test_timestamps)
        y_array = np.concatenate([sf for sf in sf_generator])
        y_hat_dict = module.predict(dc=dc, timestamps=valid_test_timestamps)

        # convert y and y_hat values to torch.Tensor
        y = torch.from_numpy(y_array)
        y_hat = torch.from_numpy(np.array(list(y_hat_dict.values())))

    # calculate metrics
    metrics = Metrics(criterion=module.trainer.criterion, y=y, y_hat=y_hat)
    print(round(metrics.objective, 4))
    print(metrics.accuracy)
    print(metrics.precision(label=0))
    print(metrics.precision(label=1))


if __name__ == '__main__':
    main()
