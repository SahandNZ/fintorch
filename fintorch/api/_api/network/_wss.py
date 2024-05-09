from typing import Dict

from ._protocol import Protocol


class Wss(Protocol):
    def __init__(self, base_url: str, proxies: Dict[str, str]):
        super().__init__(base_url=base_url, proxies=proxies)
