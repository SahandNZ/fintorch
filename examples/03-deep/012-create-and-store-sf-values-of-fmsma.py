from rich.progress import Progress

from fintorch.deep.transform.label import ForwardMiddleSmaLabelTransform
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.function import call_with_dict


def main():
    args = DefaultArgumentParser.parse()

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # create sf values
    transform = call_with_dict(ForwardMiddleSmaLabelTransform, args.transform_kwargs)
    with Progress(*RICH_PROGRESS_COLUMNS) as progress, transform:
        transform.prepare_sf(dc=dc, progress=progress)


if __name__ == '__main__':
    main()
