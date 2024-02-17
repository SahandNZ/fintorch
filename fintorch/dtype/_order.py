from datetime import datetime

from ..enum import OrderSide, OrderStatus, OrderType


class Order:
    def __init__(self):
        self.id: str = None
        self.symbol: str = None
        self.timestamp: int = None
        self.status: OrderStatus = None
        self.side: OrderSide = None
        self.quantity: float = None
        self.price: float = None
        self.stop_price: float = None

        self.activated_timestamp: int = None
        self.activated_price: float = None
        self.filled_timestamp: int = None
        self.filled_price: float = None
        self.cancel_timestamp: int = None

    @property
    def quote_asset(self) -> str:
        return self.symbol.split("-")[0]

    @property
    def base_asset(self) -> str:
        return self.symbol.split("-")[1]

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def type(self) -> OrderType:
        price_bit = 1 if self.price is not None else 0
        stop_bit = 1 if self.stop_price is not None else 0
        return OrderType(price_bit + 2 * stop_bit)

    @property
    def is_active(self) -> bool:
        return self.activated_price is not None

    @property
    def close_timestamp(self) -> int:
        return self.filled_timestamp or self.cancel_timestamp

    def __str__(self):
        return ("Order {} {} {} {} {} {}"
                .format(self.symbol, self.datetime, self.side, self.quantity, self.price, self.stop_price))
