from abc import ABC
from typing import Dict


class Protocol(ABC):
    def __init__(self, base_url: str, proxies: Dict[str, str]):
        self._base_url: str = base_url
        self._proxies = proxies or {}
