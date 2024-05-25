from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])
    with Progress(*RICH_PROGRESS_COLUMNS) as progress, args.label_transform:
        args.label_transform.prepare_sf(dc=dc, progress=progress)


if __name__ == '__main__':
    main()
