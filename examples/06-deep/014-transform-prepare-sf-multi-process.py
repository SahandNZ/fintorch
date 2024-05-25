import atexit
from multiprocessing import Process
from typing import List

from rich.progress import Progress

from fintorch.deep.transform import Transform
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser, DefaultNamespace


def terminate_processes(processes: List[Process]):
    for process in processes:
        process.terminate()


def task_target(transform: Transform, args: DefaultNamespace):
    dc = args.online_exchange.future.data.get_data_collection(
        symbols=[transform.symbol],
        time_frames=[transform.time_frame]
    )

    # create sf values
    with transform:
        transform.prepare_sf(dc=dc)


def main():
    args = DefaultArgumentParser.parse()

    # create processes
    processes = []
    for transform in args.transforms:
        process_args = (transform, args)
        process = Process(target=task_target, args=process_args)
        processes.append(process)

    # set at exit callback to terminate all processes
    atexit.register(terminate_processes, processes=processes)

    # start processes and update progress bars
    pending_process_set = set(processes)
    running_process_set = set()
    done_process_set = set()
    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        overall_task = progress.add_task(description="Overall", total=len(args.transforms))
        while len(processes) != len(done_process_set):
            # start process if there is free worker (processor)
            for process in pending_process_set:
                if len(running_process_set) < args.max_workers:
                    running_process_set.add(process)
                    process.start()

            # remove running processes from pending set
            pending_process_set = pending_process_set - running_process_set

            # update progress bars
            for process in running_process_set:
                if not process.is_alive():
                    progress.update(overall_task, advance=1)
                    done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set


if __name__ == '__main__':
    main()
