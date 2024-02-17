from typing import Union

from ..enum import PositionSide, PositionType


class Position:
    def __init__(self):
        self.symbol: str = None
        self.leverage: int = None
        self.type: PositionType = None
        self.quantity: float = None
        self.entry_timestamp: int = None
        self.entry_price: float = None
        self.current_price: float = None

        self.exit_timestamp: int = None
        self.exit_price: float = None

    @property
    def side(self) -> Union[PositionSide, None]:
        return PositionSide.LONG if self.quantity > 0 else (PositionSide.SHORT if self.quantity < 0 else None)

    @property
    def is_open(self) -> bool:
        return self.exit_price is not None

    @property
    def margin(self) -> float:
        return round(abs(self.quantity * self.entry_price), 2)

    @property
    def realized_profit(self) -> Union[float, None]:
        if not self.is_open:
            exit_price = self.exit_price
            return round((exit_price / self.entry_price - 1) * int(self.side) * self.leverage, 2)

    @property
    def unrealized_profit(self) -> Union[float, None]:
        if self.is_open:
            exit_price = self.current_price
            return round((exit_price / self.entry_price - 1) * int(self.side) * self.leverage, 2)

    def __str__(self):
        return "Position {} {} {} {}".format(self.symbol, self.side, self.quantity, self.entry_price)
