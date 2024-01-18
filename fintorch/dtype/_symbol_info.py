class SymbolInfo:
    def __init__(self):
        self.symbol: str = None
        self.base_asset: str = None
        self.quote_asset: str = None
        self.on_board_timestamp: int = None
        self.price_precision: int = None
        self.price_step: int = None
        self.volume_precision: int = None
        self.volume_step: int = None
