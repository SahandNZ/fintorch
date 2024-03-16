from abc import ABC

from ._https import Https
from ._wss import Wss


class Network(ABC):
    def __init__(self, https: Https, wss: Wss):
        self._https: Https = https
        self._wss: Wss = wss
