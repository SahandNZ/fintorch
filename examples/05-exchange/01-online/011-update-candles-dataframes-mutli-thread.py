import random
from threading import Thread

from rich.progress import Progress

from fintorch.exchange import OnlineExchange
from fintorch.defaults import RICH_PROGRESS_COLUMNS
from fintorch.utils.args import DefaultArgumentParser

def target(online_exchange: OnlineExchange, symbol: str, progress: Progress):
    online_exchange.future.data.update_candles_dataframe(
        symbol=symbol,
        progress=progress
    )

def main():
    string_args = ["--symbols-config", "all", "--max-workers", "6"]
    args = DefaultArgumentParser.parse(args=string_args)

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        pending_threads = set()
        running_threads = set()
        done_threads = set()

        for symbol in args.symbols:
            thread_args = (args.online_exchange, symbol, progress)
            thread = Thread(target=target, args=thread_args)
            pending_threads.add(thread)

        overall_task = progress.add_task("Overall", total=len(args.symbols))
        while len(done_threads) < len(args.symbols):
            if len(running_threads) < args.max_workers and 0 < len(pending_threads):
                thread = pending_threads.pop()
                running_threads.add(thread)
                thread.start()

            pending_threads = pending_threads - running_threads

            for thread in running_threads:
                if not thread.is_alive():
                    progress.update(task_id=overall_task, advance=1)
                    done_threads.add(thread)

            running_threads = running_threads - done_threads



if __name__ == '__main__':
    main()
