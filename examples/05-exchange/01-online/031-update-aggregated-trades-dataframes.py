from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        args.online_exchange.future.data.update_aggregated_trades_dataframe(
            symbol=args.symbol,
            progress=progress
        )


if __name__ == '__main__':
    main()
