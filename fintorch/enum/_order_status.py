from enum import Enum


class OrderStatus(Enum):
    OPEN = 1
    PARTIALLY_FILLED = 2
    FILLED = 4
    CANCELED = 8

    def __str__(self):
        return self.name.lower()
