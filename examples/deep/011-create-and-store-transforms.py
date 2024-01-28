import argparse
import atexit
import itertools
import json
import math
import os
import time
from datetime import datetime
from multiprocessing import Process, active_children
from typing import List, Dict

from fintorch.data import Data
from fintorch.data import load_data
from fintorch.deep.transform import Transform
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.utils.console import *
from fintorch.utils.timestamp import create_timestamps


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def task(p_index: int, console_row: int, transform_: Transform, data: Data, timestamps: List[int]):
    # define progress logs variables
    refresh_count = 10000
    refresh_rate = math.ceil(len(timestamps) / refresh_count)
    start_time = time.time()

    # create features and store them on storage
    for index, timestamp in enumerate(timestamps):
        transform_.load_or_transform(data=data, timestamp=timestamp, symbols=data.symbols, time_frames=data.time_frames)

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
                  "creating values of {:<12} | " \
                  "progress: ({:<8}/{:<8}) {} | " \
                  "speed: {} | " \
                  "elapsed: {} | " \
                  "remaining: {} | " \
                  "total: {}" \
                .format(p_index,
                        transform_.short_name,
                        index + 1, len(timestamps), progress_str,
                        speed_str,
                        elapsed_time_str,
                        remaining_time_str,
                        total_time_str)

            print_console(x=console_row, y=0, text=log)


def run_multi_process(args, symbols: List[str], time_frames: List[int], transforms: List[Transform]):
    # load candlestick data
    data = load_data(symbols=symbols, time_frames=time_frames)

    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, time_frame=args.time_frame)

    # clear console
    clear_console()

    # create customized processes pool
    child_processes = []
    child_process_to_console_row: Dict = {}
    free_console_rows = set(i for i in range(1, 48))

    # set at exit callback to terminal all child processes
    atexit.register(terminate_child_processes, child_processes)

    for process_index, transform_ in enumerate(transforms):
        # update free console rows
        for child_process in child_processes:
            console_row = child_process_to_console_row[child_process]
            if child_process.is_alive() and console_row in free_console_rows:
                free_console_rows.remove(console_row)
            elif not child_process.is_alive() and console_row not in free_console_rows:
                free_console_rows.add(console_row)

        # create child process and start it
        console_row = min(free_console_rows)
        child_process_args = (process_index + 1, console_row, transform_, data, timestamps)
        child_process = Process(target=task, args=child_process_args)
        child_process.start()

        # store child process info
        child_processes.append(child_process)
        child_process_to_console_row[child_process] = console_row

        # wait if maximum number of child processes reached
        while True:
            time.sleep(0.1)
            active_childern_count = len(active_children())
            if active_childern_count < args.processes:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--processes", action="store", type=int, required=False, default=os.cpu_count())
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    transforms = [
        # define feature transforms
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length),

        # define feature transforms
        ForwardBackwardMinimumLabelTransform(),
        ForwardIchimokuLabelTransform(),
        ForwardMiddleSmaLabelTransform(),
        ForwardRocLabelTransform(),
        NextFractalLabelTransform(),
        UpDownLabelTransform()
    ]

    run_multi_process(args=args, symbols=symbols, time_frames=time_frames, transforms=transforms)


if __name__ == '__main__':
    main()
