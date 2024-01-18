from enum import IntEnum


class PositionSide(IntEnum):
    LONG = 1
    SHORT = -1

    def reverse(self):
        return PositionSide.SHORT if 1 == self.value else PositionSide.LONG

    def __str__(self):
        return self.name.lower()
