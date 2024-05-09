from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task("Overall", total=len(args.symbols))
        for symbol in args.symbols:
            args.online_exchange.future.data.update_candles_dataframe(
                symbol=symbol,
                progress=progress
            )

            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
