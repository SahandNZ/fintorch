import gc
import time
from datetime import datetime
from multiprocessing import Queue, Process

from rich.progress import Progress

from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.engine import SimulationEngine
from fintorch.utils.args import DefaultArgumentParser


def target(queue: Queue, engine: SimulationEngine):
    try:
        with engine:
            for timestamp in engine.simulate():
                queue.put(timestamp)

        queue.put(0)

    except Exception as e:
        queue.put(str(e))
        time.sleep(20)
        queue.put(-1)


def main():
    string_args = [
        "--logger-config", "info",
        "--max-workers", "8",
    ]
    args = DefaultArgumentParser.parse(args=string_args)

    progress = Progress(*RICH_PROGRESS_COLUMNS)
    overall_task_id = progress.add_task(description="overall", total=len(args.engines))

    # multi process life cycles and properties
    pending_process_set = set()
    running_process_set = set()
    done_process_set = set()
    process_to_task_id = {}
    process_to_args = {}

    # Create pending processes and add them to set
    for engine in args.engines:
        queue = Queue()
        process_args = (queue, engine)
        process = Process(target=target, args=process_args)
        process_to_args[process] = process_args
        pending_process_set.add(process)

    # Update main_layout to track processes
    with progress:
        while len(done_process_set) < len(args.engines):
            # start process if there is free worker and work to do
            if len(running_process_set) < args.max_workers and 0 < len(pending_process_set):
                process = pending_process_set.pop()
                queue, engine = process_to_args[process]
                with engine:
                    task_id = progress.add_task(str(engine), total=len(engine.clock))

                running_process_set.add(process)
                process_to_task_id[process] = task_id

                process.start()

            # remove running processes from pending process set
            pending_process_set = pending_process_set - running_process_set

            # update terminal live display
            for process in running_process_set:
                queue, engine = process_to_args[process]
                task_id = process_to_task_id[process]
                if not queue.empty():
                    message = queue.get()
                    if isinstance(message, str):
                        print(message)
                    if isinstance(message, int):
                        message = int(message)
                        if 0 < message:
                            description = f"{str(engine)} - {str(datetime.fromtimestamp(message))}"
                            progress.update(task_id=task_id, advance=1, description=description)
                        elif 0 == message:
                            progress.update(task_id=overall_task_id, advance=1)
                            progress.update(task_id=task_id, visible=False)
                            done_process_set.add(process)
                        else:
                            progress.update(task_id=overall_task_id, advance=1)
                            progress.update(task_id=task_id, visible=False)
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
