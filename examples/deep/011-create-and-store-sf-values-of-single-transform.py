import argparse
import atexit
import itertools
import json
import time
from multiprocessing import Process, Queue
from typing import List

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.data import load_data
from fintorch.deep.transform import Transform
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.function import call_with_dict
from fintorch.utils.timestamp import create_timestamps


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    transform = RollingMeanStdTrRocFeatureTransform(symbol=args.symbol, time_frame=args.time_frame,
                                                    dim_sequence=args.dim_sequence)

    data = load_data(symbols=[transform.symbol], time_frames=[transform.time_frame])
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Creating SF values of {}".format(str(transform)), total=len(timestamps))
        for _ in transform.transform_sf(data=data, timestamps=timestamps):
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
