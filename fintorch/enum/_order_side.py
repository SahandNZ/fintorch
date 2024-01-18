from enum import IntEnum


class OrderSide(IntEnum):
    BUY = 1
    SELL = -1

    def reverse(self):
        return OrderSide.SELL if 1 == self.value else OrderSide.BUY

    def __str__(self):
        return self.name.lower()
