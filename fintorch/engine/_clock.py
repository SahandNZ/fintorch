import os
import math
import pickle
import filelock
from datetime import datetime
from typing import Dict, Union

from ..enum import TimeFrame
from ..utils.timestamp import floor_timestamp, ceil_timestamp, to_timestamp


class Clock:
    def __init__(self, start_date: Union[str, datetime], stop_date: Union[str, datetime], interval: TimeFrame):
        self.__start_timestamp: int = to_timestamp(date=start_date)
        self.__stop_timestamp: int = to_timestamp(date=stop_date) if stop_date is not None else None
        self.__interval: TimeFrame = interval

        # state
        self.__timestamp: Union[int, None] = None

    @property
    def start_timestamp(self) -> int:
        return self.__start_timestamp

    @property
    def stop_timestamp(self) -> int:
        return self.__stop_timestamp

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.stop_timestamp)

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    def next(self) -> Union[int, None]:
        current_timestamp = int(datetime.now().timestamp())
        current_open_timestamp = floor_timestamp(timestamp=current_timestamp, time_frame=self.interval)
        last_closed_timestamp = current_open_timestamp - self.interval
        stop_timestamp = self.stop_timestamp or math.inf
        stop_timestamp = min(stop_timestamp, last_closed_timestamp)
        
        if self.timestamp < stop_timestamp:
            self.__timestamp = self.timestamp + self.interval
            return self.timestamp

    def state_dict(self) -> Dict:
        state_dict = {"timestamp": self.timestamp}
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        self.__timestamp = state_dict.get("timestamp", self.start_timestamp)

    def open(self, directory: str) -> None:
        path = os.path.join(directory, "clock.pkl")
        try:
            with open(path, "rb") as file:
                state_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            state_dict = {}

        self.load_state_dict(state_dict=state_dict)

    def close(self, directory: str) -> None:
        path = os.path.join(directory, "clock.pkl")
        with open(path, "wb+") as file:
            pickle.dump(self.state_dict(), file)

    def __len__(self):
        current_timestamp = int(datetime.now().timestamp())
        current_open_timestamp = ceil_timestamp(timestamp=current_timestamp, time_frame=self.interval)
        return math.floor((current_open_timestamp - self.timestamp) / self.interval)
