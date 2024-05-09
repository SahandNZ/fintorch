from datetime import datetime
from typing import Union


class Ticker:
    def __init__(self):
        self.symbol: Union[str, None] = None
        self.timestamp: Union[float, None] = None
        self.price: Union[float, None] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None
