from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    string_args = ["--symbols-config", "5"]
    args = DefaultArgumentParser.parse(args=string_args)

    # optimize modules with rich panel
    overall_progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task = overall_progress.add_task(description="overall jobs", total=len(args.modules))
    progress_panel = Panel.fit(overall_progress, title="Overall progress")

    with Live(refresh_per_second=2) as live:
        for module in args.modules:
            # load data collection
            dc = args.online_exchange.future.data.get_data_collection(
                symbols=[module.symbol],
                time_frames=module.dataset.time_frames
            )

            # optimize folds
            with module:
                for status in module.optimize(dc=dc, start_date=args.start_date):
                    live.update(Group(Panel.fit(str(status), title=str(module)), progress_panel))

            overall_progress.update(overall_task, advance=1)


if __name__ == '__main__':
    main()
