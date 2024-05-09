import gc
import time
import warnings
from multiprocessing import Queue, Process
from typing import List

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from fintorch.deep.module import Module
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser, DefaultNamespace


def target(queue: Queue, module: Module, args: DefaultNamespace):
    try:
        dc = args.online_exchange.future.data.get_data_collection(
            symbols=[module.symbol],
            time_frames=[module.time_frame]
        )

        with module:
            for status in module.optimize(dc=dc, start_date=args.start_date):
                queue.put(str(status))

    except Exception as e:
        queue.put(str(e))
        time.sleep(20)

    queue.put(-1)


def main():
    warnings.filterwarnings("ignore")
    args = DefaultArgumentParser.parse()

    # create rich main layout
    rows_count, columns_count = 5, 5

    overall_progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task = overall_progress.add_task(description="overall", total=len(args.modules))
    progress_panel = Panel.fit(overall_progress, title="Overall")

    rows = [Layout(name=f"row-{i}") for i in range(rows_count)]
    rows.append(Layout(progress_panel, name="progress"))

    main_layout = Layout()
    main_layout.split_column(*rows)
    free_layouts: List[Layout] = []
    for i in range(rows_count):
        layouts = [Layout(Panel("Pending..."), name=f"col-{j}") for j in range(columns_count)]
        main_layout[f"row-{i}"].split_row(*layouts)
        free_layouts.extend(layouts)

    # multi process life cycles and properties
    pending_process_set = set()
    running_process_set = set()
    done_process_set = set()
    process_to_layout = {}
    process_to_args = {}

    # Create pending processes and add them to set
    for module in args.modules:
        queue = Queue()
        process = Process(target=target, args=(queue, module, args))
        process_to_args[process] = (queue, module)
        pending_process_set.add(process)

    # Update main_layout to track processes
    with Live(main_layout, refresh_per_second=2):
        while len(done_process_set) < len(args.modules):
            # start process if there is free worker and work to do
            if len(running_process_set) < args.max_workers and 0 < len(pending_process_set):
                process = pending_process_set.pop()
                process_to_layout[process] = free_layouts.pop(0)
                running_process_set.add(process)
                process.start()

            # remove running processes from pending process set
            pending_process_set = pending_process_set - running_process_set

            # update terminal live display
            for process in running_process_set:
                queue, module = process_to_args[process]
                layout = process_to_layout[process]
                if not queue.empty():
                    message = queue.get()
                    if isinstance(message, str):
                        layout.update(Panel(message, title=f"{module}"))
                    elif isinstance(message, int) and -1 == message:
                        overall_progress.update(task_id=overall_task, advance=1)
                        layout.update(Panel("Pending..."))
                        free_layouts.append(layout)
                        done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set

            # remove args of done processes
            for process in done_process_set:
                if process in process_to_args:
                    del process_to_args[process]
                    gc.collect()


if __name__ == '__main__':
    main()
