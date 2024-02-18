import argparse
import atexit
import itertools
import json
import time
from datetime import datetime
from multiprocessing import Process, Queue
from typing import List

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform import Transform
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.enum import TimeFrame
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.function import call_with_dict
from fintorch.utils.timestamp import create_timestamps


def terminate_processes(processes: List[Process]):
    for process in processes:
        process.terminate()


def task_target(transform: Transform):
    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[transform.symbol], time_frames=[transform.time_frame])

    # create timestamps
    symbol_info = dc.get_symbol_info(symbol=transform.symbol)
    on_board_timestamp = symbol_info.on_board_timestamp
    current_timestamp = int(datetime.now().timestamp() // int(transform.time_frame) * int(transform.time_frame))
    timestamps = list(range(on_board_timestamp, current_timestamp, int(transform.time_frame)))

    # create sf values
    sf_generator = transform.transform_sf(dc=dc, timestamps=timestamps)
    for _ in sf_generator:
        pass


def run_multi_process(args, transforms: List[Transform]):
    pending_process_set = set()
    running_process_set = set()
    done_process_set = set()

    # create processes
    for transform in transforms:
        process_args = (transform, )
        process = Process(target=task_target, args=process_args)
        pending_process_set.add(process)

    # set at exit callback to terminate all processes
    atexit.register(terminate_processes, processes=list(pending_process_set))

    # start processes and update progress bars
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        total_items = len(transforms)
        overall_task = progress.add_task(description="Overall", total=total_items)
        while 0 < len(pending_process_set) or 0 < len(running_process_set):
            # start process if there is free worker (processor)
            for process in pending_process_set:
                alive_process_count = sum(process.is_alive() for process in pending_process_set)
                if alive_process_count < args.max_workers:
                    running_process_set.add(process)
                    process.start()

            # remove running processes from pending set
            pending_process_set = pending_process_set - running_process_set

            # update progress bars
            for process in running_process_set:
                if not process.is_alive():
                    progress.update(overall_task, advance=1)
                    done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # get symbols and time_frames fom config_dict
    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    transform_types = [
        RollingMeanStdTrRocFeatureTransform,
        StftTrRocFeatureTransform,

        ForwardBackwardMinimumLabelTransform,
        ForwardIchimokuLabelTransform,
        ForwardMiddleSmaLabelTransform,
        ForwardRocLabelTransform,
        NextFractalLabelTransform,
        UpDownLabelTransform
    ]

    transforms = []
    for transform_type, symbol, time_frame in itertools.product(transform_types, symbols, time_frames):
        kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
        transform = call_with_dict(transform_type, kwargs)
        transforms.append(transform)

    run_multi_process(args=args, transforms=transforms)


if __name__ == '__main__':
    main()
