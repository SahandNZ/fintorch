from enum import IntEnum


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1
    STOP_MARKET = 2
    STOP_LIMIT = 3

    @property
    def is_stop(self) -> bool:
        return OrderType.STOP_MARKET == self or OrderType.STOP_LIMIT == self

    @property
    def is_market(self) -> bool:
        return OrderType.MARKET == self or OrderType.STOP_MARKET == self

    def __str__(self):
        return self.name.lower().replace("_", "-")
