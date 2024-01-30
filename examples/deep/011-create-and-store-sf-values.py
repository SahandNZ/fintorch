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
from fintorch.utils.timestamp import create_timestamps


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def task_target(transform_: Transform, symbol: str, time_frame: int, timestamps: List[int], queue: Queue):
    # load candlestick data
    data = load_data(symbols=[symbol], time_frames=[time_frame])

    previous_message_time, previous_message_index = time.time(), 0
    for index, timestamp in enumerate(timestamps):
        transform_.load_or_transform_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)
        if 1 < time.time() - previous_message_time:
            queue.put((index - previous_message_index + 1))
            previous_message_time = time.time()
            previous_message_index = index

    queue.put((-1))


def run_multi_process(args, symbols: List[str], time_frames: List[int], transforms: List[Transform]):
    # set at exit callback to terminate all processes
    atexit.register(terminate_child_processes)

    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    # create processes
    process_to_args = {}
    for transform, symbol, time_frame in itertools.product(transforms, symbols, time_frames):
        queue = Queue()
        process_args = (transform, symbol, time_frame, timestamps, queue)
        process = Process(target=task_target, args=process_args)
        process_to_args[process] = (transform, symbol, time_frame, queue)

    # start processes and update progress bars
    process_to_task = {}
    process_to_is_started = {p: False for p in process_to_args.keys()}
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        total_items = len(transforms) * len(symbols) * len(time_frames) * len(timestamps)
        total_task = progress.add_task(description="Total", total=total_items)

        while any(process.is_alive() or not is_started for process, is_started in process_to_is_started.items()):
            for process, (transform, symbol, time_frame, queue) in process_to_args.items():
                # start process if there is free worker (processor)
                is_started = process_to_is_started[process]
                alive_process_count = sum(process.is_alive() for process in process_to_args.keys())
                if not is_started and alive_process_count < args.max_workers:
                    process_to_is_started[process] = True
                    process.start()

                # create progress bar
                if process not in process_to_task and process.is_alive() and process_to_is_started[process]:
                    description = "Creating and storing SF values of {:^12} for {}-{}" \
                        .format(transform.short_name, symbol, time_frame)
                    task = progress.add_task(description=description, total=len(timestamps))
                    process_to_task[process] = task

                # update progress bar
                if not queue.empty():
                    advance = queue.get()
                    task = process_to_task[process]
                    if 0 < advance:
                        progress.update(task, advance=advance)
                        progress.update(total_task, advance=advance)
                    else:
                        progress.update(task, visible=False)
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
    transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.dim_sequence),
        StftTrRocFeatureTransform(sequence_length=args.dim_sequence),

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
