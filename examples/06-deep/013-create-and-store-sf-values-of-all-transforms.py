from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=args.symbols, time_frames=args.time_frames)
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task(description="Overall", total=len(args.transforms))
        for transform in args.transforms:
            # create sf values
            with transform:
                transform.prepare_sf(dc=dc, progress=progress)

            # update progress bar
            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
