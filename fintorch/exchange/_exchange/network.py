from abc import ABC

from .https import Https
from .wss import Wss


class Network(ABC):
    def __init__(self, https: Https, wss: Wss):
        self._https: Https = https
        self._wss: Wss = wss
