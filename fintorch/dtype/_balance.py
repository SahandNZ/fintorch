class Balance:
    def __init__(self):
        self.asset: str = None
        self.total: float = None
        self.frozen: float = None

    @property
    def available(self) -> float:
        return self.total - self.frozen
