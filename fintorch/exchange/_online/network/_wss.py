from abc import abstractmethod
from typing import Dict, Callable, Any

from fintorch.exchange._online.network._protocol import Protocol
from fintorch.utils.websocket import Websocket


class Wss(Protocol):
    def __init__(self, base_url: str, key: str, secret_key: str, proxies: Dict[str, str]):
        super().__init__(base_url, key, secret_key, proxies)
        socks5 = self._proxies.get("socks5", None)
        self.__ws: Websocket = Websocket(url=base_url, on_message=self._on_message, socks5=socks5)

    def _send(self, payload: str):
        self.__ws.send(payload)

    @abstractmethod
    def _on_message(self, message: str):
        raise NotImplementedError()

    @abstractmethod
    def subscribe_stream(self, stream: str, on_message: Callable[[Dict], Any]):
        raise NotImplementedError()

    def join(self):
        self.__ws.join()
