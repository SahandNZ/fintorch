from datetime import datetime
from typing import List

import pandas as pd


class Ticker:
    def __init__(self):
        self.symbol: str = None
        self.timestamp: int = None
        self.price: float = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None