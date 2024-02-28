from rich.progress import Progress

from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS, BASE_TIME_FRAME
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task("Overall", total=len(args.symbols))
        for symbol in args.symbols:
            ONLINE_EXCHANGE.future.data.update_candles_dataframe(
                symbol=symbol,
                time_frame=BASE_TIME_FRAME,
                progress=progress
            )

            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
