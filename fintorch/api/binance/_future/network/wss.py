from typing import Dict

from ...._api.network import Wss


class BinanceFutureWss(Wss):
    def __init__(self, api_key: str = None, secret_key: str = None, proxies: Dict[str, str] = None):
        super().__init__(base_url="", proxies=proxies)
        self.__api_key: str = api_key
        self.__secret_key: str = secret_key
