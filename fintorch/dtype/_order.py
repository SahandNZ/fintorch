from datetime import datetime
from typing import Union

from ..enum import OrderSide, OrderStatus, OrderType


class Order:
    def __init__(self):
        self.symbol: Union[str, None] = None
        self.side: Union[OrderSide, None] = None
        self.percentage: Union[float, None] = None
        self.reduce_only: Union[bool, None] = None
        self.price: Union[float, None] = None
        self.stop_price: Union[float, None] = None
        self.comment: Union[bool, None] = None

        self.id: Union[str, None] = None
        self.timestamp: Union[int, None] = None
        self.activated_timestamp: Union[float, None] = None
        self.filled_timestamp: Union[float, None] = None
        self.canceled_timestamp: Union[float, None] = None

        self.status: Union[OrderStatus, None] = None
        self.activated_price: Union[float, None] = None
        self.filled_price: Union[float, None] = None

        self.quantity: Union[float, None] = None

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
    def activated_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.activated_timestamp) if self.activated_timestamp is not None else None

    @property
    def is_activated(self) -> bool:
        return self.activated_price is not None

    @property
    def filled_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.filled_timestamp) if self.filled_timestamp is not None else None

    @property
    def cancel_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.canceled_timestamp) if self.canceled_timestamp is not None else None

    @property
    def close_timestamp(self) -> float:
        return self.filled_timestamp or self.canceled_timestamp

    @property
    def close_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.close_timestamp) if self.close_timestamp is not None else None

    def __str__(self):
        return ("Order {} {}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                "\t- {:<32}{}\n"
                .format(self.symbol, str(self.type),
                        "Date", self.datetime,
                        "Side", self.side,
                        "Quantity", self.quantity,
                        "Price", self.price,
                        "Stop Price", self.stop_price,
                        "Activated Date", self.activated_datetime,
                        "Fill Date", self.filled_datetime,
                        "Fill Price", self.filled_price,
                        "Cancel Date", self.cancel_datetime))
