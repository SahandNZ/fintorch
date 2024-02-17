import argparse
from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform.feature import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    transform = RollingMeanStdTrRocFeatureTransform(
        symbol=args.symbol,
        time_frame=args.time_frame,
        dim_sequence=args.dim_sequence
    )

    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[transform.symbol], time_frames=[transform.time_frame])

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Creating SF values of {}".format(str(transform)), total=len(timestamps))
        for _ in transform.transform_sf(dc=dc, timestamps=timestamps):
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
