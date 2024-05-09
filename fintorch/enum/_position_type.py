from enum import IntEnum


class PositionType(IntEnum):
    ISOLATED = 1
    CROSS = 2

    def __str__(self):
        return self.name.lower()
