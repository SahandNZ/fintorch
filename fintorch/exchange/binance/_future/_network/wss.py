import json
from typing import Dict, Callable, Any

from ...._online import Wss


class BinanceFutureWss(Wss):
    def __init__(self, key: str = None, secret_key: str = None, proxies: Dict[str, str] = None):
        super().__init__(base_url="wss://fstream.binance.com/stream", key=key, secret_key=secret_key, proxies=proxies)

        self._stream_to_on_message: Dict[str, Callable[[str], Any]] = {}

    def _on_message(self, message: str):
        message_dict = json.loads(message)
        if "stream" in message_dict:
            stream = message_dict["stream"]
            data = message_dict["exchange"]

            if stream in self._stream_to_on_message:
                stream_on_message = self._stream_to_on_message[stream]
                stream_on_message(data)

    def subscribe_stream(self, stream: str, on_message: Callable[[Dict], Any]):
        stream_id = len(self._stream_to_on_message)
        payload_dict = {"method": "SUBSCRIBE", "params": [stream], "id": stream_id}
        payload = json.dumps(payload_dict)

        self._send(payload)
        self._stream_to_on_message[stream] = on_message
