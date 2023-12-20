import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Type

from pyccx.data import load_dataframes_dict
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, \
    SpinnerColumn, MofNCompleteColumn

from fintorch.data import Data
from fintorch.transform.feature.rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.feature.stft_tr_roc import StftTrRocFeatureTransform

exchange: str = "binance"
max_workers: int = None
progress_bar_columns: List = None


def work(symbol: str, time_frame: int, feature_transform_cls: Type, progress: Progress):
    symbols = [symbol]
    time_frames = [time_frame]
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames, update=False,
                                   progress=progress)
    data = Data(df_dict)

    timestamps = data[symbol, time_frame].index.to_list()
    feature_transform = feature_transform_cls(symbols=symbols, time_frames=time_frames)
    feature_transform.fit_transform(data=data, timestamps=timestamps, progress=progress)


def run_multi_thread(items: List):
    with Progress(*progress_bar_columns) as progress:
        main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for symbol, time_frame, feature_transform_cls in items:
                future = executor.submit(work, symbol, time_frame, feature_transform_cls, progress)
                futures.append(future)

            for future in futures:
                future.result()
                progress.update(main_task, advance=1)


def run_multi_process(items: List):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for symbol, time_frame, feature_transform_cls in items:
            future = executor.submit(work, symbol, time_frame, feature_transform_cls, None)
            futures.append(future)

        with Progress(*progress_bar_columns) as progress:
            main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

            for future in futures:
                future.result()
                progress.update(main_task, advance=1)


def run_sequential(items: List):
    with Progress(*progress_bar_columns) as progress:
        main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

        for symbol, time_frame, feature_transform_cls in items:
            work(symbol, time_frame, feature_transform_cls, progress)
            progress.update(main_task, advance=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-thread", action="store_true", required=False)
    parser.add_argument("--multi-process", action="store_true", required=False)
    parser.add_argument("--max-workers", action="store", type=int, required=False, default=16)
    parser.add_argument("--config-path", action="store", type=str, required=False, default="config/datasets.json")
    args = parser.parse_args()

    # set values of global variables
    global max_workers, progress_bar_columns
    max_workers = args.max_workers
    progress_bar_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature transforms
    feature_transforms_cls = [
        RollingMeanStdTrRocFeatureTransform,
        StftTrRocFeatureTransform
    ]

    # run works
    items = list(itertools.product(symbols, time_frames, feature_transforms_cls))
    run_func = run_multi_thread if args.multi_thread else (run_multi_process if args.multi_process else run_sequential)
    run_func(items=items)


if __name__ == '__main__':
    main()
