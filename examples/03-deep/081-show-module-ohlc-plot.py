from fintorch.deep.model import FeedForward
from fintorch.deep.module import create_default_module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    # define module
    module = create_default_module(
        feature_transform_type=RollingMeanStdTrRocFeatureTransform,
        label_transform_type=ForwardMiddleSmaLabelTransform,
        model_type=FeedForward,
    )

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])

    # show ohlc plot
    with module:
        module.show_ohlc_plot(dc=dc, start_date=args.start_date, stop_date=args.stop_date)


if __name__ == '__main__':
    main()
