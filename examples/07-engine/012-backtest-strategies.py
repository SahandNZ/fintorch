from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser


def main():
    string_args = ["--logger-config", "info"]
    args = DefaultArgumentParser.parse(args=string_args)

    progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task_id = progress.add_task(description="overall", total=len(args.engines))

    with progress:
        for engine in args.engines:
            with engine:
                if 0 < len(engine.clock):
                    task_id = progress.add_task(description=str(engine), total=len(engine.clock))
                    for timestamp in engine.simulate():
                        progress.update(task_id=task_id, advance=1)
                    progress.update(task_id=task_id, visible=False)

                progress.update(task_id=overall_task_id, advance=1)


if __name__ == '__main__':
    main()
