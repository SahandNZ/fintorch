from enum import Enum


class MarketType(Enum):
    SPOT = 1
    FUTURE = 2

    def __str__(self):
        return self.name.lower()
