import argparse
import itertools
import json
from datetime import datetime

from rich.progress import Progress

from examples.args import add_default_args_and_parse
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.function import call_with_dict
from fintorch.utils.timestamp import create_timestamps


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

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=symbols, time_frames=time_frames)

    # define transforms
    items = list(itertools.product(transform_types, symbols, time_frames))

    # create sf values
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task(description="Overall", total=len(items))
        for transform_type, symbol, time_frame in items:
            kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
            transform = call_with_dict(transform_type, kwargs)

            # create timestamps
            symbol_info = dc.get_symbol_info(symbol=symbol)
            on_board_timestamp = symbol_info.on_board_timestamp
            current_timestamp = int(datetime.now().timestamp() // int(time_frame) * int(time_frame))
            timestamps = list(range(on_board_timestamp, current_timestamp, int(time_frame)))

            # create sub task
            description = f"Creating sf values of {str(transform)} {symbol} {str(time_frame)}"
            sub_task = progress.add_task(description=description, total=len(timestamps))

            # create sf values
            sf_generator = transform.transform_sf(dc=dc, timestamps=timestamps)
            for _ in sf_generator:
                progress.update(task_id=sub_task, advance=1)

            # update progress bar
            progress.update(task_id=sub_task, visible=False)
            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
