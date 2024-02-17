from abc import ABC
from typing import Dict


class Protocol(ABC):
    def __init__(self, base_url: str, key: str, secret_key: str, proxies: Dict[str, str]):
        self._base_url: str = base_url
        self._key: str = key
        self._secret_key: str = secret_key
        self._proxies = proxies or {}
