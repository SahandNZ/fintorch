from enum import IntEnum


class OrderStatus(IntEnum):
    OPEN = 1
    PARTIALLY_FILLED = 2
    FILLED = 4
    CANCELED = 8

    @property
    def is_closed(self) -> bool:
        return OrderStatus.OPEN != self.value and OrderStatus.PARTIALLY_FILLED != self.value

    def __str__(self):
        return self.name.lower()
