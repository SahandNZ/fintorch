import argparse
import atexit
import itertools
import json
import math
import os
import time
from datetime import datetime
from multiprocessing import Process
from typing import List, Dict

from rich.progress import Progress

from fintorch.data import load_data
from fintorch.deep.dtype import Dataset
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.dtype import Data
from fintorch.enum import TimeFrame
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.console import print_console, clear_console


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def task(p_index: int, console_row: int, start_date: str, stop_date: str, sampling_time_frame: TimeFrame, data: Data,
         sequence_length: int, feature_transform: FeatureTransform, label_transform: LabelTransform):
    # create dataset and fit data to it
    dataset = Dataset(
        start_date=start_date,
        stop_date=stop_date,
        sampling_time_frame=sampling_time_frame,
        symbols=data.symbols,
        time_frames=data.time_frames,
        sequence_length=sequence_length,
        feature_transform=feature_transform,
        label_transform=label_transform
    )
    dataset.fit(data=data)

    # define progress logs variables
    refresh_count = 1000
    refresh_rate = math.ceil(len(dataset.timestamps) / refresh_count)
    start_time = time.time()

    # create features and store them on storage under the hood
    for index, timestamp in enumerate(dataset.timestamps):
        dataset.load_sample(timestamp=timestamp)

        # progress logs
        if 0 == (index + 1) % refresh_rate or (index + 1) == len(dataset.timestamps):
            progress = (index + 1) / len(dataset.timestamps) * 100
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
                  "creating samples | " \
                  "progress: ({:<8}/{:<8}) {} | " \
                  "speed: {} | " \
                  "elapsed: {} | " \
                  "remaining: {} | " \
                  "total: {}" \
                .format(p_index,
                        index + 1, len(dataset.timestamps), progress_str,
                        speed_str,
                        elapsed_time_str,
                        remaining_time_str,
                        total_time_str)

            print_console(x=console_row, y=0, text=log)


def run_multi_process(args, data: Data, feature_transforms: List[FeatureTransform],
                      label_transforms: List[LabelTransform]):
    # clear console
    clear_console()

    # create customized processes pool
    child_processes = []
    child_process_to_console_row: Dict = {}
    free_console_rows = set(i for i in range(1, 65))

    # set at exit callback to terminal all child processes
    atexit.register(terminate_child_processes, child_processes)

    items = list(itertools.product(feature_transforms, label_transforms))
    for p_index, (ft, lt) in enumerate(items):
        # update free console rows
        for child_process in child_processes:
            console_row = child_process_to_console_row[child_process]
            if child_process.is_alive() and console_row in free_console_rows:
                free_console_rows.remove(console_row)
            elif not child_process.is_alive() and console_row not in free_console_rows:
                free_console_rows.add(console_row)

        # create child process and start it
        console_row = min(free_console_rows)

        child_process_args = (p_index + 1, console_row, args.start_date, args.stop_date, args.sampling_time_frame,
                              data, args.sequence_length, ft, lt)
        child_process = Process(target=task, args=child_process_args)
        child_process.start()

        # store child process info
        child_processes.append(child_process)
        child_process_to_console_row[child_process] = console_row

        # wait if maximum number of child processes reached
        while True:
            time.sleep(0.1)
            alive_process_count = sum(p.is_alive() for p in child_processes)
            if alive_process_count < args.process_count:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--process-count", action="store", type=int, required=False, default=os.cpu_count())
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.sequence_length),
        StftTrRocFeatureTransform(sequence_length=args.sequence_length)
    ]

    label_transforms = [
        ForwardBackwardMinimumLabelTransform(),
        ForwardIchimokuLabelTransform(),
        ForwardMiddleSmaLabelTransform(),
        ForwardRocLabelTransform(),
        NextFractalLabelTransform(),
        UpDownLabelTransform()
    ]

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        data = load_data(symbols=symbols, time_frames=time_frames, progress=progress)

    run_multi_process(
        args=args,
        data=data,
        feature_transforms=feature_transforms,
        label_transforms=label_transforms
    )


if __name__ == '__main__':
    main()
