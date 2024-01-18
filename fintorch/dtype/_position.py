from ..enum import PositionSide, PositionType


class Position:
    def __init__(self):
        self.symbol: str = None
        self.side: PositionSide = None
        self.type: PositionType = None
        self.margin: float = None
        self.volume: float = None
        self.entry_price: float = None
        self.profit: float = None
        self.leverage: int = None
        self.liquidation_price: float = None
