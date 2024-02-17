from abc import ABC

from fintorch.exchange._online.network._https import Https
from fintorch.exchange._online.network._wss import Wss


class Network(ABC):
    def __init__(self, https: Https, wss: Wss):
        self._https: Https = https
        self._wss: Wss = wss
