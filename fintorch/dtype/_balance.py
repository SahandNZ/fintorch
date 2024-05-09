from typing import Union


class Balance:
    def __init__(self):
        self.asset: Union[str, None] = None
        self.available: Union[float, None] = None
