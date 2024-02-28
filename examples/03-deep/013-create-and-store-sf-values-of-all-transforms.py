import itertools

from rich.progress import Progress

from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser
from fintorch.utils.function import call_with_dict


def main():
    args = DefaultArgumentParser.parse()

    # define feature and label transforms
    transform_types = FEATURE_TRANSFORM_TYPES + LABEL_TRANSFORM_TYPES

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=args.symbols, time_frames=args.time_frames)

    # define transforms
    items = list(itertools.product(transform_types, args.symbols, args.time_frames))
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task(description="Overall", total=len(items))
        for transform_type, symbol, time_frame in items:
            transform_kwargs = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
            transform = call_with_dict(transform_type, transform_kwargs)

            # create sf values
            with transform:
                transform.prepare_valid_sf(dc=dc, progress=progress)

            # update progress bar
            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
