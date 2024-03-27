from rich.live import Live
from rich.panel import Panel

from fintorch.deep.model import *
from fintorch.deep.transform.label import *
from fintorch.deep.transform.feature import *
from fintorch.deep.module import create_module_from_args
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

    # optimize folds
    with Live(refresh_per_second=2) as live, module:
        for status in module.optimize(dc=dc, start_date=args.start_date):
            live.update(Panel.fit(str(status), title=f"{str(module)}"))


if __name__ == '__main__':
    main()
