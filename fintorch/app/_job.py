from datetime import datetime
from typing import Callable, Dict
from typing import List

from apscheduler.job import Job as APSJob

from ._context import Context


class Job:
    def __init__(self, callback: Callable):
        self.__callback: callback = callback

        self.__aps_job: APSJob = None
        self.__removed: bool = False
        self.__enabled: bool = False

    @property
    def name(self) -> str:
        return self.callback.__name__

    @property
    def callback(self) -> Callable:
        return self.__callback

    @property
    def aps_job(self) -> APSJob:
        return self.__aps_job

    @aps_job.setter
    def aps_job(self, job: APSJob):
        self.__aps_job = job

    @property
    def removed(self) -> bool:
        return self.__removed

    @property
    def enabled(self) -> bool:
        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value:
            self.aps_job.resume()
        else:
            self.aps_job.pause()
        self.__enabled = value

    @property
    def next_run_time(self) -> datetime:
        return self.aps_job.next_run_time

    def schedule_removal(self) -> None:
        self.aps_job.remove()
        self.__removed = True

    def run(self, context: Context, args: List, kwargs: Dict) -> None:
        context.refresh()
        self.callback(context, *args, **kwargs)
