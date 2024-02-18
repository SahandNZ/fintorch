import argparse
import atexit
import itertools
import json
from multiprocessing import Process
from typing import List

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform import Transform
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.function import call_with_dict


def terminate_processes(processes: List[Process]):
    for process in processes:
        process.terminate()


def task_target(transform: Transform):
    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[transform.symbol], time_frames=[transform.time_frame])

    # create sf values
    transform.prepare_sf(dc=dc)


def run_multi_process(args, transforms: List[Transform]):
    # create processes
    processes = []
    for transform in transforms:
        process_args = (transform,)
        process = Process(target=task_target, args=process_args)
        processes.append(process)

    # set at exit callback to terminate all processes
    atexit.register(terminate_processes, processes=processes)

    # start processes and update progress bars
    pending_process_set = set(processes)
    running_process_set = set()
    done_process_set = set()
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        total_items = len(transforms)
        overall_task = progress.add_task(description="Overall", total=total_items)
        while len(processes) != len(done_process_set):
            # start process if there is free worker (processor)
            for process in pending_process_set:
                if len(running_process_set) < args.max_workers:
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
    transform_types = FEATURE_TRANSFORM_TYPES + LABEL_TRANSFORM_TYPES

    transforms = []
    for transform_type, symbol, time_frame in itertools.product(transform_types, symbols, time_frames):
        transform_kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
        transform = call_with_dict(transform_type, transform_kwargs)
        transforms.append(transform)

    run_multi_process(args=args, transforms=transforms)


if __name__ == '__main__':
    main()
