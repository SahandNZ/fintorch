import copy
import ssl
import time
from threading import Lock, Thread
from typing import List, Dict, Callable, Any
from urllib.parse import urlparse

import websocket


class Websocket:
    def __init__(self, url: str, on_message: Callable[[str], Any], socks5: str = None):
        self._url: str = url
        self._on_message: Callable[[str], Any] = on_message
        self._socks5 = urlparse(socks5) if socks5 is not None else None

        self._ssl_opt = {"cert_reqs": ssl.CERT_NONE}
        if self._socks5 is None:
            self._proxy_params: Dict = {}
        else:
            self._proxy_params: Dict = {
                "proxy_type": "socks5",
                "http_proxy_host": self._socks5.hostname,
                "http_proxy_port": self._socks5.port,
                "http_proxy_auth": (self._socks5.username, self._socks5.password)
            }

        self._ws = websocket.WebSocket(sslopt=self._ssl_opt)
        self._send_history: List[str] = []
        self._send_buffer: List[str] = []

        self._send_lock: Lock = Lock()
        self._send_thread: Thread = Thread(target=self._send_loop)
        self._recv_thread: Thread = Thread(target=self._recv_loop)

        self._send_thread.start()
        self._recv_thread.start()

    def send(self, payload: str):
        with self._send_lock:
            self._send_history.append(payload)
            self._send_buffer.append(payload)

    def join(self):
        self._send_thread.join()
        self._recv_thread.join()

    def _connect(self):
        with self._send_lock:
            if not self._ws.connected and 0 < len(self._send_history):
                self._ws.connect(self._url, **self._proxy_params)
                self._send_buffer = copy.deepcopy(self._send_history)

    def _send(self):
        with self._send_lock:
            if self._ws.connected and 0 < len(self._send_buffer):
                message = self._send_buffer.pop(0)
                self._ws.send(message)
                time.sleep(0.05)

    def _recv(self):
        if self._ws.connected:
            message = self._ws.recv()
            if 0 < len(message):
                self._on_message(message)

    def _send_loop(self):
        while True:
            try:
                self._connect()
                self._send()
            except Exception as e:
                print("send", e)

    def _recv_loop(self):
        while True:
            try:
                self._recv()
            except Exception as e:
                print("recv:", e)
