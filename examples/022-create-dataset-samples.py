import argparse
import atexit
import json
import math
import time
from datetime import datetime
from multiprocessing import Process
from typing import List, Dict

from pyccx.data import load_dataframes_dict
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.transform.feature import *
from fintorch.transform.label import *
from fintorch.utils.console import print_console, clear_console

EXCHANGE: str = "binance"


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def create_timestamps_divisions(args) -> List[int]:
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    stop_date = datetime.strptime(args.stop_date, "%Y-%m-%d")

    # create timestamps
    start_timestamp = math.ceil(start_date.timestamp() / args.sampling_time_frame) * args.sampling_time_frame
    stop_timestamp = math.floor(stop_date.timestamp() / args.sampling_time_frame) * args.sampling_time_frame
    timestamps = list(range(start_timestamp, stop_timestamp, args.sampling_time_frame))

    # create divisions
    divisions = []
    step = math.ceil(len(timestamps) / args.divisions_count)
    for start_index in range(0, len(timestamps), step):
        stop_index = start_index + step
        division = timestamps[start_index: stop_index]
        divisions.append(division)

    return divisions


def task(p_index: int, console_row: int, dataset: Dataset, timestamps: List[int]):
    # define progress logs variables
    refresh_count = 100
    refresh_rate = math.ceil(len(timestamps) / refresh_count)
    start_time = time.time()

    # create features and store them on storage under the hood
    for index, timestamp in enumerate(timestamps):
        dataset.create_sample(timestamp=timestamp)

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
                  "creating samples | " \
                  "progress: ({:<8}/{:<8}) {} | " \
                  "speed: {} | " \
                  "elapsed: {} | " \
                  "remaining: {} | " \
                  "total: {}" \
                .format(p_index,
                        index, len(timestamps), progress_str,
                        speed_str,
                        elapsed_time_str,
                        remaining_time_str,
                        total_time_str)

            print_console(x=console_row, y=0, text=log)


def run_multi_process(args, dataset: Dataset):
    # create and divide timestamps
    timestamps_divisions = create_timestamps_divisions(args=args)

    # clear console
    clear_console()

    # create customized processes pool
    child_processes = []
    child_process_to_console_row: Dict = {}
    free_console_rows = set(i for i in range(1, 65))

    # set at exit callback to terminal all child processes
    atexit.register(terminate_child_processes, child_processes)

    for process_index, timestamps_division in enumerate(timestamps_divisions):
        # update free console rows
        for child_process in child_processes:
            console_row = child_process_to_console_row[child_process]
            if child_process.is_alive() and console_row in free_console_rows:
                free_console_rows.remove(console_row)
            elif not child_process.is_alive() and console_row not in free_console_rows:
                free_console_rows.add(console_row)

        # create child process and start it
        console_row = min(free_console_rows)
        child_process_args = (process_index + 1, console_row, dataset, timestamps_division)
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
    parser.add_argument("--processes", action="store", type=int, required=False, default=24)
    parser.add_argument("--divisions-count", action="store", type=int, required=False, default=96)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=2 ** 8)
    parser.add_argument("--sampling-time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/config.json")
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

    # create dataset
    dataset = Dataset(
        symbols=symbols,
        time_frames=time_frames,
        sequence_length=args.sequence_length,
        sampling_time_frame=args.sampling_time_frame,
        feature_transforms=feature_transforms,
        label_transforms=label_transforms
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        # load candlestick data
        df_dict = load_dataframes_dict(EXCHANGE, symbols, time_frames, progress=progress)
        data = Data(df_dict)

        # fit data to dataset
        dataset.fit(data=data, progress=progress)

    run_multi_process(args=args, dataset=dataset)


if __name__ == '__main__':
    main()
