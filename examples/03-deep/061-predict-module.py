import time

from fintorch.deep.model import *
from fintorch.deep.module import create_module_from_args
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    # define module
    module = create_module_from_args(
        feature_transform_type=RollingMeanStdTrRocFeatureTransform,
        label_transform_type=ForwardMiddleSmaLabelTransform,
        model_type=FeedForward
    )

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])
    test_timestamps = module.get_timestamps(dc=dc)

    start_time = time.time()
    with module:
        y_hat_dict = module.predict(dc=dc, timestamps=test_timestamps)
    elapsed_time_ms = (time.time() - start_time) * 1000
    print("Loading predictions takes: {:.3f} ms".format(elapsed_time_ms))
    
    print(sum(y_hat is None for y_hat in y_hat_dict.values()))


if __name__ == '__main__':
    main()
