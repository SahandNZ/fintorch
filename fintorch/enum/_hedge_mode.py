from enum import Enum


class HedgeMode(Enum):
    ONE_WAY = 1
    TWO_WAY = 2

    def __str__(self):
        return self.name.lower()
