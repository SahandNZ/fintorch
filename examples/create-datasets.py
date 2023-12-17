import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Type

from pyccx.data import load_dataframes_dict
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, \
    SpinnerColumn, MofNCompleteColumn

from fintorch.data import Data
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.transform.feature.rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.feature.stft_tr_roc import StftTrRocFeatureTransform
from fintorch.transform.label.classification.trend.up_down import UpDownLabelTransform


def work(exchange: str, symbol: str, time_frame: int, feature_transform_cls: Type, progress: Progress):
    symbols = [symbol]
    time_frames = [time_frame]

    # load candlestick data
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames, update=False,
                                   show_progress_bar=False)
    data = Data(df_dict)

    # create and store features for all timestamp
    label_transform = UpDownLabelTransform(symbol=symbol, time_frame=time_frame)
    feature_transform = feature_transform_cls(symbols=symbols, time_frames=time_frames)
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)
    dataset.prepare(data=data, progress=progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-thread", action="store_true", required=False)
    parser.add_argument("--multi-process", action="store_true", required=False)
    parser.add_argument("--max-workers", action="store", type=int, required=False, default=16)
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--config-path", action="store", type=str, required=False, default="datasets.json")
    args = parser.parse_args()

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
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    if args.multi_thread:
        with Progress(*progress_columns) as progress:
            main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = []
                for symbol, time_frame, feature_transform_cls in items:
                    future = executor.submit(work, args.exchange, symbol, time_frame, feature_transform_cls, progress)
                    futures.append(future)

                for future in futures:
                    future.result()
                    progress.update(main_task, advance=1)

    elif args.multi_process:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for symbol, time_frame, feature_transform_cls in items:
                future = executor.submit(work, args.exchange, symbol, time_frame, feature_transform_cls, None)
                futures.append(future)

            with Progress(*progress_columns) as progress:
                main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

                for future in futures:
                    future.result()
                    progress.update(main_task, advance=1)

    else:
        with Progress(*progress_columns) as progress:
            main_task = progress.add_task(description="[red]Creating datasets", total=len(items))

            for symbol, time_frame, feature_transform_cls in items:
                work(args.exchange, symbol, time_frame, feature_transform_cls, progress)
                progress.update(main_task, advance=1)


if __name__ == '__main__':
    main()
