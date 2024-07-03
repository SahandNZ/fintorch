import pickle
import filelock
from abc import ABC
from datetime import datetime
from typing import Dict, Union

from fintorch.enum import MarketType


class MarketElement(ABC):
    def __init__(self, market_type: MarketType):
        self.__market_type: MarketType = market_type

        # states
        self.__timestamp: Union[int, None] = None

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    def state_dict(self) -> Dict:
        return {"timestamp": self.timestamp}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.__timestamp = state_dict.get("timestamp")
        
    def clear_state_dict(self) -> None:
        self.__timestamp = None

    def open(self, path: str):
        try:
            with open(path, "rb") as file:
                state_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            state_dict = {}

        self.load_state_dict(state_dict=state_dict)

    def next(self, timestamp: int) -> None:
        self.__timestamp = timestamp

    def close(self, path: str):
        with open(path, "wb+") as file:
            pickle.dump(self.state_dict(), file)
            
        self.clear_state_dict()
