from abc import ABC

from .network import Https, Network, Wss
from .. import Wallet


class OnlineWallet(Network, Wallet, ABC):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https=https, wss=wss)
