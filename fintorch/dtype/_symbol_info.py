from datetime import datetime
from typing import Union


class SymbolInfo:
    def __init__(self):
        self.symbol: Union[str, None] = None
        self.base_asset: Union[str, None] = None
        self.quote_asset: Union[str, None] = None
        self.on_board_timestamp: Union[float, None] = None
        self.price_precision: Union[int, None] = None
        self.price_step: Union[int, None] = None
        self.quantity_precision: Union[int, None] = None
        self.quantity_step: Union[int, None] = None

    @property
    def on_board_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.on_board_timestamp) if self.on_board_timestamp is not None else None
