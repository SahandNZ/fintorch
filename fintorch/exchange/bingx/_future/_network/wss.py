from typing import Any, Callable, Dict

from ...._online import Wss


class BingxFutureWss(Wss):
    def __init__(self, base_url: str, key: str, secret_key: str, proxies: Dict[str, str]):
        super().__init__(base_url, key, secret_key, proxies)

    def _on_message(self, message: str):
        raise NotImplementedError()

    def subscribe_stream(self, stream: str, on_message: Callable[[Dict], Any]):
        raise NotImplementedError()
