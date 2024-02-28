import time

from fintorch.deep.model import FeedForward
from fintorch.deep.module import create_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.timestamp import create_timestamps


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

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    start_time = time.time()
    with module:
        y_hat_dict = module.predict(dc=dc, timestamps=timestamps)
    elapsed_time_ms = (time.time() - start_time) * 1000
    print("Loading predictions takes: {:.3f} ms".format(elapsed_time_ms))


if __name__ == '__main__':
    main()
