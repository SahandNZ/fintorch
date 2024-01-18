from datetime import datetime

from ..enum import OrderSide, OrderStatus, OrderType


class Order:
    def __init__(self):
        self.id: str = None
        self.symbol: str = None
        self.timestamp: int = None
        self.datetime: datetime = None
        self.type: OrderType = None
        self.status: OrderStatus = None
        self.side: OrderSide = None
        self.volume: float = None
        self.price: float = None
        self.stop_price: float = None
