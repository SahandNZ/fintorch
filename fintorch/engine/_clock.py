import math
import time
from datetime import datetime
from typing import Iterator

from fintorch.enum import TimeFrame


class Clock:
    def __init__(self, interval: TimeFrame, speed: int):
        self.__interval: TimeFrame = interval
        self.__speed: int = speed

        self.__start_timestamp: int = None
        self.__stop_timestamp: int = None
        self.__timestamp: int = None

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    @property
    def speed(self) -> int:
        return self.__speed

    @property
    def simulated_wait_time(self) -> float:
        return self.interval / self.speed if self.speed is None else 0

    @property
    def start_timestamp(self) -> int:
        return self.__start_timestamp

    @property
    def stop_timestamp(self) -> int:
        return self.__stop_timestamp

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    def __call__(self, start_date: str, stop_date: str = None) -> Iterator:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(stop_date, str):
            stop_date = datetime.strptime(stop_date, "%Y-%m-%d")

        self.__start_timestamp = start_date.timestamp()
        self.__stop_timestamp = stop_date.timestamp() if stop_date is not None else None

        return self.__iter__()

    def __len__(self):
        return math.floor((self.stop_timestamp - self.start_timestamp) / self.interval)

    def __iter__(self):
        self.__timestamp = self.start_timestamp - self.interval
        return self

    def __next__(self):
        self.__timestamp += self.interval
        # used in backtest mode
        if self.stop_timestamp is not None:
            if self.timestamp < self.stop_timestamp:
                time.sleep(self.simulated_wait_time)
                return int(self.timestamp)
            else:
                raise StopIteration()

        # used in livetest mode
        else:
            current_timestamp = datetime.now().timestamp()
            current_open_timestamp = current_timestamp // int(self.interval) * int(self.interval)
            if self.timestamp < current_open_timestamp:
                time.sleep(self.simulated_wait_time)
                return int(self.timestamp)
            else:
                next_open_timestamp = current_open_timestamp + int(self.interval)
                wait_time = math.ceil(next_open_timestamp - current_timestamp)
                time.sleep(wait_time)
                return int(self.timestamp)
