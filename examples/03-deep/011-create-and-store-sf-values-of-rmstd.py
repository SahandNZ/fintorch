import argparse
from datetime import datetime

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform.feature import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    transform = RollingMeanStdTrRocFeatureTransform(
        symbol=args.symbol,
        time_frame=args.time_frame,
        dim_sequence=args.dim_sequence
    )

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[transform.symbol], time_frames=[transform.time_frame])

    # create sf values
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        transform.prepare_sf(dc=dc, progress=progress)


if __name__ == '__main__':
    main()
