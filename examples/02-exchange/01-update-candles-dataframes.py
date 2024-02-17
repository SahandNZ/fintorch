import argparse
import json
from typing import List

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import BASE_TIME_FRAME, RICH_PROGRESS_COLUMNS


def work(symbol: str, progress: Progress):
    ONLINE_EXCHANGE.future.data.update_candles_dataframe(symbol=symbol, time_frame=BASE_TIME_FRAME, progress=progress)


def run(symbols: List[str]):
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        for symbol in symbols:
            work(symbol=symbol, progress=progress)


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    run(symbols=config_dict["symbols"])


if __name__ == '__main__':
    main()
