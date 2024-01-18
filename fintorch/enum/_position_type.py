from enum import Enum


class PositionType(Enum):
    ISOLATED = 1
    CROSS = 2

    def __str__(self):
        return self.name.lower()
