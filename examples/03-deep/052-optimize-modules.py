from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from fintorch.deep.module import create_modules_from_args
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    modules = create_modules_from_args(args=args)

    # optimize modules with rich panel
    overall_progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task = overall_progress.add_task(description="overall jobs", total=len(modules))
    progress_panel = Panel.fit(overall_progress, title="Overall progress")

    with Live(refresh_per_second=2) as live:
        for module in modules:
            # load data collection
            dc = ONLINE_EXCHANGE.future.data.get_data_collection(
                symbols=[module.symbol],
                time_frames=[module.time_frame]
            )

            # optimize folds
            with module:
                for status in module.optimize(dc=dc, start_date=args.start_date):
                    live.update(Group(Panel.fit(str(status), title=str(module)), progress_panel))

            overall_progress.update(overall_task, advance=1)


if __name__ == '__main__':
    main()
