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


def terminate_processes(processes: List[Process]):
    for process in processes:
        process.terminate()


def task_target(transform: Transform, timestamps: List[int], queue: Queue):
    data = load_data(symbols=[transform.symbol], time_frames=[transform.time_frame])
    sf_generator = transform.transform_sf(data, timestamps)

    previous_time, previous_index = time.time(), 0
    for index, sf in enumerate(sf_generator):
        if 1 < time.time() - previous_time:
            queue.put(index - previous_index)
            previous_time, previous_index = time.time(), index

    queue.put(-1)


def run_multi_process(args, transforms: List[Transform]):
    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    # create processes
    process_to_args = {}
    for transform in transforms:
        queue = Queue()
        process_args = (transform, timestamps, queue)
        process = Process(target=task_target, args=process_args)
        process_to_args[process] = (transform, queue)

    # set at exit callback to terminate all processes
    atexit.register(terminate_processes, processes=list(process_to_args.keys()))

    # start processes and update progress bars
    process_to_task = {}
    pending_process_set = set(process_to_args.keys())
    running_process_set = set()
    done_process_set = set()
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        total_items = len(transforms) * len(timestamps)
        overall_task = progress.add_task(description="Overall", total=total_items)
        while 0 < len(pending_process_set) or 0 < len(running_process_set):
            # start process if there is free worker (processor)
            for process in pending_process_set:
                alive_process_count = sum(process.is_alive() for process in process_to_args.keys())
                if alive_process_count < args.max_workers:
                    running_process_set.add(process)
                    process.start()

            # remove running processes from pending set
            pending_process_set = pending_process_set - running_process_set

            # update progress bars
            for process in running_process_set:
                transform, queue = process_to_args[process]
                if process not in process_to_task:
                    description = "Creating SF values of ({})".format(transform)
                    task = progress.add_task(description=description, total=len(timestamps))
                    process_to_task[process] = task

                # update progress bar
                if not queue.empty():
                    advance = queue.get()
                    task = process_to_task[process]
                    if 0 < advance:
                        progress.update(task, advance=advance)
                        progress.update(overall_task, advance=advance)
                    else:
                        progress.update(task, visible=False)
                        done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set

            # remove args of done processes
            for process in done_process_set:
                if process in process_to_args:
                    del process_to_args[process]


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
