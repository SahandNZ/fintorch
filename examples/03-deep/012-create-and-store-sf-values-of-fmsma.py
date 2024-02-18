import argparse
from datetime import datetime

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform.label import ForwardMiddleSmaLabelTransform
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    transform = ForwardMiddleSmaLabelTransform(symbol=args.symbol, time_frame=args.time_frame)

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[transform.symbol], time_frames=[transform.time_frame])

    # create timestamps
    symbol_info = dc.get_symbol_info(symbol=args.symbol)
    on_board_timestamp = symbol_info.on_board_timestamp
    current_timestamp = int(datetime.now().timestamp() // args.interval * args.interval)
    timestamps = list(range(on_board_timestamp, current_timestamp, args.interval))

    # create sf values
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(description="Creating SF values of {}".format(str(transform)), total=len(timestamps))
        for _ in transform.transform_sf(dc=dc, timestamps=timestamps):
            progress.update(task, advance=1)


if __name__ == '__main__':
    main()
