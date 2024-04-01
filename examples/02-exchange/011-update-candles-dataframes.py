import itertools

from rich.progress import Progress

from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS, BASE_TIME_FRAME
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        items = list(itertools.product(args.symbols, args.time_frames))
        overall_task = progress.add_task("Overall", total=len(items))
        for symbol, time_frame in items:
            ONLINE_EXCHANGE.future.data.update_candles_dataframe(
                symbol=symbol,
                time_frame=time_frame,
                progress=progress
            )

            progress.update(task_id=overall_task, advance=1)


if __name__ == '__main__':
    main()
