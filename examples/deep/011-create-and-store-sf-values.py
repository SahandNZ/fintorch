import argparse
import itertools
import json
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


def target(transform_: Transform, symbol: str, time_frame: int, timestamps: List[int], queue: Queue):
    # load candlestick data
    data = load_data(symbols=[symbol], time_frames=[time_frame])

    for index, timestamp in enumerate(timestamps):
        transform_.load_or_transform_sf(data=data, timestamp=timestamp, symbol=symbol, time_frame=time_frame)
        if (index + 1) % 100 == 0:
            queue.put((100))
    queue.put((-1))


def run_multi_process(args, symbols: List[str], time_frames: List[int], transforms: List[Transform]):
    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, interval=args.interval)

    queues_dict = {}
    subprocesses_dict = {}
    for key in itertools.product(transforms, symbols, time_frames):
        queue = Queue()
        queues_dict[key] = queue

        subprocess_args = (*key, timestamps, queue)
        subprocesses = Process(target=target, args=subprocess_args)
        subprocesses_dict[key] = (subprocesses, False)

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        subprocess_index = 0
        tasks_dict = {}
        while any(subprocess.is_alive() or not is_started for subprocess, is_started in subprocesses_dict.values()):
            while sum(subprocess.is_alive() for subprocess, _ in subprocesses_dict.values()) < args.max_workers:
                if len(subprocesses_dict) <= subprocess_index:
                    break

                key, (subprocess, is_started) = list(subprocesses_dict.items())[subprocess_index]
                transform, symbol, time_frame = key
                subprocess_index += 1

                # start subprocess
                subprocess.start()
                subprocesses_dict[key] = (subprocess, True)

                # create task in rich progress bar
                description = "Creating and storing SF values of {:^12} for {}-{}" \
                    .format(transform.short_name, symbol, time_frame)
                task = progress.add_task(description=description, total=len(timestamps))
                tasks_dict[key] = task

            # update rich progress bars
            for key, (subprocess, is_started) in subprocesses_dict.items():
                if is_started:
                    queue = queues_dict[key]
                    if not queue.empty():
                        advance = queue.get()
                        task = tasks_dict[key]
                        if 0 < advance:
                            progress.update(task, advance=advance)
                        else:
                            progress.update(task, visible=False)


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
