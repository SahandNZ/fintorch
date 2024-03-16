from datetime import datetime


class SymbolInfo:
    def __init__(self):
        self.symbol: str = None
        self.base_asset: str = None
        self.quote_asset: str = None
        self.on_board_timestamp: int = None
        self.price_precision: int = None
        self.price_step: int = None
        self.quantity_precision: int = None
        self.quantity_step: int = None

    @property
    def on_board_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.on_board_timestamp) if self.on_board_timestamp is not None else None
