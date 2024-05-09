from enum import IntEnum


class OrderStatus(IntEnum):
    OPEN = 1
    PARTIALLY_FILLED = 2
    FILLED = 4
    CANCELED = 8

    def __str__(self):
        return self.name.lower()
