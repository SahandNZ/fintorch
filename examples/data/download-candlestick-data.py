import argparse
import json
from typing import List

from rich.progress import Progress

from fintorch.data import LOCAL_DATA
from fintorch.setting import RICH_PROGRESS_COLUMNS


def work(args, symbol: str, progress: Progress):
    LOCAL_DATA.download_candles(symbol=symbol, time_frame=args.time_frame, progress=progress)


def run(args, symbols: List[str]):
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        for symbol in symbols:
            work(args, symbol, progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=900)
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    run(args=args, symbols=config_dict["symbols"])


if __name__ == '__main__':
    main()
