import math
import os
import pickle
from datetime import datetime
from typing import Dict, Union

from ..enum import TimeFrame
from ..exchange import OnlineExchange
from ..strategy import Strategy
from ..utils.timestamp import to_timestamp, floor_timestamp, ceil_timestamp


class Clock:
    def __init__(self, online_exchange: OnlineExchange, strategy: Strategy, interval: TimeFrame):
        self.__interval: TimeFrame = interval

        symbol_info = online_exchange.future.data.get_symbol_info(symbol=strategy.symbol)
        self.__start_timestamp: float = symbol_info.on_board_timestamp
        self.__start_timestamp: float = to_timestamp(date="2021-01-01")

        # state
        self.__timestamp: Union[float, None] = None

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    @property
    def start_timestamp(self) -> float:
        return self.__start_timestamp

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.start_timestamp)

    @property
    def timestamp(self) -> float:
        return self.__timestamp

    def next(self) -> Union[float, None]:
        current_timestamp = datetime.now().timestamp()
        current_open_timestamp = floor_timestamp(timestamp=current_timestamp, time_frame=self.interval)
        if self.timestamp < current_open_timestamp:
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
