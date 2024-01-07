import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List

from pyccx.data import load_dataframes_dict
from rich.progress import Progress

from fintorch.data import Data
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.transform.feature import *
from fintorch.utils.console import *


def create_timestamps(start_date: str, stop_date: str, time_frame: int) -> List[int]:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    stop_date = datetime.strptime(stop_date, "%Y-%m-%d")

    start_timestamp = math.ceil(start_date.timestamp() / time_frame) * time_frame
    stop_timestamp = math.floor(stop_date.timestamp() / time_frame) * time_frame

    return list(range(start_timestamp, stop_timestamp + 1, time_frame))


def divide_timestamps(timestamps: List[int], divisions_count: int) -> List[List[int]]:
    divisions = []
    step = math.ceil(len(timestamps) / divisions_count)
    for start_index in range(0, len(timestamps), step):
        stop_index = start_index + step
        division = timestamps[start_index: stop_index]
        divisions.append(division)

    return divisions


def work(work_index: int, data: Data, timestamps: List[int], feature_transform: FeatureTransform):
    # define progress logs variables
    refresh_count = 1000
    refresh_rate = int(len(timestamps) / refresh_count)
    start_time = time.time()

    # fit data to feature transform
    feature_transform.fit(data=data)

    # create features and store them on storage under the hood
    for index, timestamp in enumerate(timestamps):
        feature_transform.transform(timestamp=timestamp)

        # progress logs
        if 0 == (index + 1) % refresh_rate or (index + 1) == len(timestamps):
            progress = (index + 1) / len(timestamps) * 100
            elapsed_time = time.time() - start_time
            speed = elapsed_time / (index + 1) if (index + 1) < elapsed_time else (index + 1) / elapsed_time
            speed_unit = "sec/iter" if (index + 1) < elapsed_time else "iter/sec"
            total_time = elapsed_time * 100 / progress
            remaining_time = total_time - elapsed_time

            progress_str = "{:<6.2f}%".format(progress)
            speed_str = "{:<6.1f} {}".format(speed, speed_unit)
            elapsed_time_str = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
            total_time_str = datetime.strftime(datetime.utcfromtimestamp(total_time), '%H:%M:%S')
            remaining_time_str = datetime.strftime(datetime.utcfromtimestamp(remaining_time), '%H:%M:%S')

            log = "Process #{:<3} | " \
                  "creating features of {:<5} | " \
                  "progress: {} | " \
                  "speed: {} | " \
                  "elapsed: {} | " \
                  "remaining: {} | " \
                  "total: {}" \
                .format(work_index,
                        feature_transform.short_name,
                        progress_str,
                        speed_str,
                        elapsed_time_str,
                        remaining_time_str,
                        total_time_str)

            print_console(x=work_index, y=0, text=log)


def run_multi_process(args, symbols: List[str], time_frames: List[int]):
    # define feature transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    ]

    # load candlestick data
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        df_dict = load_dataframes_dict(args.exchange, symbols, time_frames, progress=progress)
        data = Data(df_dict)

    # divide timestamps to create more sub processes
    divisions_count = int(args.works / len(feature_transforms))
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, time_frame=args.time_frame)
    timestamps_divisions = divide_timestamps(timestamps=timestamps, divisions_count=divisions_count)

    clear_console()

    # create sub processes
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for feature_transform in feature_transforms:
            for ts_division in timestamps_divisions:
                future = executor.submit(work, len(futures) + 1, data, ts_division, feature_transform)
                futures.append(future)

        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--works", action="store", type=int, required=False, default=24)
    parser.add_argument("--workers", action="store", type=int, required=False, default=24)
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=2 ** 8)
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/config.json")
    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    run_multi_process(args=args, symbols=config_dict["symbols"], time_frames=config_dict["time-frames"])


if __name__ == '__main__':
    main()
