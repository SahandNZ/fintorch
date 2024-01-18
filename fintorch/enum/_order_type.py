from enum import Enum


class OrderType(Enum):
    LIMIT = 1
    MARKET = 2

    def __str__(self):
        return self.name.lower()
