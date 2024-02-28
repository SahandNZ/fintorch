import time
from multiprocessing import Process, Queue
from typing import List

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

from fintorch.deep.module import Module, create_modules
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser


def target(module: Module, queue: Queue):
    try:
        # load data collection
        dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[module.symbol], time_frames=[module.time_frame])

        # optimize folds
        with module:
            for status in module.optimize(dc=dc):
                queue.put(str(status))
        queue.put(-1)
    except Exception as e:
        queue.put(str(e))
        time.sleep(10)
        queue.put(-1)


def create_process(modules: List[Module]):
    for module in modules:
        # create process
        process_args = (module, Queue())
        process = Process(target=target, args=process_args)

        yield process, process_args


def run_multi_process(args, modules: List[Module]):
    # create rich main layout
    rows, columns = 6, 4
    main_layout = Layout()
    main_layout.split_column(*[Layout(name=f"row-{i}") for i in range(rows)])
    free_layouts: List[Layout] = []
    for i in range(rows):
        layouts = [Layout(Panel("Pending..."), name=f"col-{j}") for j in range(columns)]
        main_layout[f"row-{i}"].split_row(*layouts)
        free_layouts.extend(layouts)

    # create process generator to avoid from too many open files issue
    process_generator = create_process(modules=modules)

    # create terminal layout to track processes
    process_to_args = {}
    process_to_layout = {}
    running_process_set = set()
    done_process_set = set()
    with Live(main_layout, refresh_per_second=2):
        while len(done_process_set) < len(modules):
            # start process if there is free worker (processor)
            if len(running_process_set) < args.max_workers:
                process, process_args = next(process_generator)
                process_to_args[process] = process_args
                process_to_layout[process] = free_layouts.pop(0)
                running_process_set.add(process)
                process.start()

            # update terminal live display
            for process in running_process_set:
                module, queue = process_to_args[process]
                layout = process_to_layout[process]
                if not queue.empty():
                    message = queue.get()
                    if isinstance(message, str):
                        layout.update(Panel(message, title=f"{module}"))
                    elif isinstance(message, int) and -1 == message:
                        layout.update(Panel("Pending..."))
                        free_layouts.append(layout)
                        done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set

            # remove args of done processes
            for process in done_process_set:
                if process in process_to_args:
                    del process_to_args[process]


def main():
    args = DefaultArgumentParser.parse()

    modules = create_modules(args=args)
    run_multi_process(args=args, modules=modules)


if __name__ == '__main__':
    main()
