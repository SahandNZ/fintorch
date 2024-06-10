from datetime import datetime
from typing import Callable, List, Dict, Union

from apscheduler.schedulers.blocking import BlockingScheduler

from ._context import Context
from ._job import Job
from ..enum import TimeFrame
from ..utils.timestamp import ceil_timestamp


class JobQueue:
    def __init__(self, context: Context, executor):
        self.__context: Context = context
        self.__scheduler = BlockingScheduler(executors={"default": executor})

    @property
    def context(self) -> Context:
        return self.__context

    @property
    def scheduler(self) -> BlockingScheduler:
        return self.__scheduler

    @staticmethod
    def _cast_args(args: List) -> List:
        if args is None:
            return []
        else:
            return list(args)

    @staticmethod
    def _cast_kwargs(kwargs):
        return kwargs if kwargs is not None else {}

    @staticmethod
    def _cast_when(interval: TimeFrame, when: str) -> Union[datetime, None]:
        if "any" == when:
            return None
        elif "open" == when:
            next_timestamp = ceil_timestamp(timestamp=datetime.now().timestamp(), time_frame=interval)
            return datetime.fromtimestamp(next_timestamp)

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()

    def run_once(
            self,
            callback: Callable,
            args: List = None,
            kwargs: Dict = None,
            refresh_context: bool = True,
            misfire_grace_time: TimeFrame = None,
    ) -> Job:
        args = self._cast_args(args)
        kwargs = self._cast_kwargs(kwargs)

        job = Job(callback=callback, refresh_context=refresh_context)
        job.aps_job = self.scheduler.add_job(
            func=job.run,
            name=job.name,
            args=(self.__context, args, kwargs),
            misfire_grace_time=int(misfire_grace_time),
        )

        return job

    def run_repeating(
            self,
            callback: Callable,
            interval: TimeFrame,
            when: str = "any",
            args: List = None,
            kwargs: Dict = None,
            refresh_context: bool = True,
            misfire_grace_time: TimeFrame = None,
    ) -> Job:
        args = self._cast_args(args)
        kwargs = self._cast_kwargs(kwargs)
        start_date = self._cast_when(interval=interval, when=when)

        job = Job(callback=callback, refresh_context=refresh_context)
        job.aps_job = self.scheduler.add_job(
            func=job.run,
            name=job.name,
            seconds=interval,
            trigger="interval",
            start_date=start_date,
            args=(self.__context, args, kwargs),
            misfire_grace_time=int(misfire_grace_time),
        )

        return job
