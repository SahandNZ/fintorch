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

            # create sf values
            transform.prepare_sf(dc=dc, progress=progress)

            # update progress bar
            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
