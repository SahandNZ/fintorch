import argparse
import json
from multiprocessing import Process, Queue
from typing import List

from rich.progress import Progress

from fintorch.data import Data
from fintorch.data import load_data
from fintorch.deep.transform import Transform
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.timestamp import create_timestamps


def terminate_child_processes(child_processes: List[Process]):
    for child_process in child_processes:
        child_process.terminate()


def target(transform_: Transform, data: Data, timestamps: List[int], queue: Queue):
    for index, timestamp in enumerate(timestamps):
        transform_.load_or_transform(data=data, timestamp=timestamp, symbols=data.symbols, time_frames=data.time_frames)
        queue.put((1))


def run_multi_process(args, symbols: List[str], time_frames: List[int], transforms: List[Transform]):
    # load candlestick data
    data = load_data(symbols=symbols, time_frames=time_frames)

    # create timestamps
    timestamps = create_timestamps(start_date=args.start_date, stop_date=args.stop_date, time_frame=args.time_frame)

    transform_to_queue = {}
    transform_to_child_process = {}
    for transform_ in transforms:
        queue = Queue()
        child_process_args = (transform_, data, timestamps, queue)
        child_process = Process(target=target, args=child_process_args)
        child_process.start()
        transform_to_queue[transform_] = queue
        transform_to_child_process[transform_] = child_process

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        transform_to_task = {}
        for transform_ in transforms:
            task = progress.add_task("Creating " + transform_.short_name, total=len(timestamps))
            transform_to_task[transform_] = task

        while any(child_process.is_alive() for child_process in transform_to_child_process.values()):
            for transform_, queue in transform_to_queue.items():
                task = transform_to_task[transform_]
                if not queue.empty():
                    advance = queue.get()
                    progress.update(task, advance=advance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--sequence-length", action="store", type=int, required=False, default=32)
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
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
