from datetime import datetime
from typing import Callable, List, Dict

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler

from ._context import Context
from ._job import Job


class JobQueue:
    def __init__(self, context: Context):
        self.__context: Context = context
        self.__scheduler: BlockingScheduler = None

    @property
    def context(self) -> Context:
        return self.__context

    @property
    def scheduler(self) -> BlockingScheduler:
        if self.__scheduler is None:
            executors = {'default': ThreadPoolExecutor(max_workers=8)}
            self.__scheduler = BlockingScheduler(executors=executors)
        return self.__scheduler

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()

    def _cast_args(self, args: List) -> List:
        if args is None:
            return []
        else:
            return list(args)

    def _cast_kwargs(self, kwargs):
        return kwargs if kwargs is not None else {}

    def _cast_when(self, interval: int, when: str) -> datetime:
        if 'any' == when:
            return None
        elif 'open' == when:
            next_timestamp = datetime.now().timestamp() // interval * interval + interval
            return datetime.fromtimestamp(next_timestamp)

    def run_once(self, callback: Callable, args: List = None, kwargs: Dict = None) -> Job:
        args = self._cast_args(args)
        kwargs = self._cast_kwargs(kwargs)

        job = Job(callback=callback)
        job.aps_job = self.scheduler.add_job(
            func=job.run,
            name=job.name,
            args=(self.__context, args, kwargs),
        )

        return job

    def run_repeating(self, callback: Callable, interval: int, args: List = None, kwargs: Dict = None,
                      when: str = 'any', misfire_grace_time: int = None) -> Job:
        args = self._cast_args(args)
        kwargs = self._cast_kwargs(kwargs)
        start_date = self._cast_when(interval=interval, when=when)

        job = Job(callback=callback)
        job.aps_job = self.scheduler.add_job(
            func=job.run,
            name=job.name,
            seconds=interval,
            trigger="interval",
            start_date=start_date,
            args=(self.__context, args, kwargs),
            misfire_grace_time=misfire_grace_time
        )

        return job
