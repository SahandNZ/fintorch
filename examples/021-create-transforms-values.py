import argparse
import atexit
import itertools
import json
import math
import time
from datetime import datetime
from multiprocessing import Process
from typing import List, Dict

from pyccx.data import load_dataframes_dict

from fintorch.data import Data
from fintorch.transform.feature import *
from fintorch.transform.label import *
from fintorch.transform.transform import Transform
from fintorch.utils.console import *

EXCHANGE: str = "binance"


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def create_timestamps(start_date: str, stop_date: str, time_frame: int) -> List[int]:
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    stop_date = datetime.strptime(stop_date, "%Y-%m-%d")

    start_timestamp = math.ceil(start_date.timestamp() / time_frame) * time_frame
    stop_timestamp = math.floor(stop_date.timestamp() / time_frame) * time_frame

    return list(range(start_timestamp, stop_timestamp + 1, time_frame))


def task(p_index: int, console_row: int, symbol: str, time_frame: int, transform_: Transform, timestamps: List[int]):
    df_dict = load_dataframes_dict(exchange=EXCHANGE, symbols=[symbol], time_frames=[time_frame])
    data = Data(df_dict)

    # define progress logs variables
    refresh_count = 1000
    refresh_rate = int(len(timestamps) / refresh_count)
    start_time = time.time()

    # fit data to feature transform
    transform_.fit(data=data)

    # create features and store them on storage under the hood
    for index, timestamp in enumerate(timestamps):
        transform_.transform(timestamp=timestamp)

        # progress logs
        if 0 == (index + 1) % refresh_rate or (index + 1) == len(timestamps):
            progress = (index + 1) / len(timestamps) * 100
            elapsed_time = time.time() - start_time
            speed = elapsed_time / (index + 1) if (index + 1) < elapsed_time else (index + 1) / elapsed_time
            speed_unit = "sec/iter" if (index + 1) < elapsed_time else "iter/sec"
            total_time = elapsed_time * 100 / progress
            remaining_time = total_time - elapsed_time

            progress_str = "{:<6.2f}%".format(progress)
            speed_str = "{:<7.1f} {}".format(speed, speed_unit)
            elapsed_time_str = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
            total_time_str = datetime.strftime(datetime.utcfromtimestamp(total_time), '%H:%M:%S')
            remaining_time_str = datetime.strftime(datetime.utcfromtimestamp(remaining_time), '%H:%M:%S')

            log = "Process #{:<6} | " \
                  "creating values of {:<12} {:<10} {:<5} | " \
                  "progress: ({:<8}/{:<8}) {} | " \
                  "speed: {} | " \
                  "elapsed: {} | " \
                  "remaining: {} | " \
                  "total: {}" \
                .format(p_index,
                        transform_.short_name, data.symbols[0], str(data.time_frames[0]),
                        index, len(timestamps), progress_str,
                        speed_str,
                        elapsed_time_str,
                        remaining_time_str,
                        total_time_str)

            print_console(x=console_row, y=0, text=log)


def run_multi_process(args, symbols: List[str], time_frames: List[int], transforms: List[Transform]):
    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, time_frame=args.time_frame)

    # clear console
    clear_console()

    # define items as args for child processes
    items = list(itertools.product(symbols, time_frames, transforms))

    # create customized processes pool
    child_processes = []
    child_process_to_console_row: Dict = {}
    free_console_rows = set(i for i in range(1, 65))

    # set at exit callback to terminal all child processes
    atexit.register(terminate_child_processes, child_processes)

    for process_index, (symbol, time_frame, transform_) in enumerate(items):
        # update free console rows
        for child_process in child_processes:
            console_row = child_process_to_console_row[child_process]
            if child_process.is_alive() and console_row in free_console_rows:
                free_console_rows.remove(console_row)
            elif not child_process.is_alive() and console_row not in free_console_rows:
                free_console_rows.add(console_row)

        # create child process and start it
        console_row = min(free_console_rows)
        child_process_args = (process_index + 1, console_row, symbol, time_frame, transform_, timestamps)
        child_process = Process(target=task, args=child_process_args)
        child_process.start()

        # store child process info
        child_processes.append(child_process)
        child_process_to_console_row[child_process] = console_row

        # wait if maximum number of child processes reached
        while True:
            time.sleep(1)
            alive_process_count = sum(p.is_alive() for p in child_processes)
            if alive_process_count < args.processes:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", action="store_true")
    parser.add_argument("--processes", action="store", type=int, required=False, default=24)
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=2 ** 8)
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/config.json")
    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # define feature transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    ]

    # define feature transforms
    label_transforms = [
        ForwardBackwardMinimumLabelTransform(),
        ForwardIchimokuLabelTransform(),
        ForwardMiddleSmaLabelTransform(),
        ForwardRocLabelTransform(),
        NextFractalLabelTransform(),
        UpDownLabelTransform()
    ]

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]
    transforms = feature_transforms if args.feature else label_transforms

    run_multi_process(args=args, symbols=symbols, time_frames=time_frames, transforms=transforms)


if __name__ == '__main__':
    main()
