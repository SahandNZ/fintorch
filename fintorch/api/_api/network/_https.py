from abc import abstractmethod
from typing import Dict, Tuple
from urllib import parse

import requests
from requests import Response

from ._protocol import Protocol


class Https(Protocol):
    def __init__(self, base_url: str, proxies: Dict[str, str]):
        super().__init__(base_url=base_url, proxies=proxies)

    @abstractmethod
    def sign(self, method: str, endpoint: str, params: Dict, timestamp: int) -> str:
        raise NotImplementedError()

    @abstractmethod
    def prepare(self, method: str, endpoint: str, params: Dict, sign: bool) -> Tuple[Dict, Dict, Dict]:
        raise NotImplementedError()

    @abstractmethod
    def parse(self, response: Response):
        raise NotImplementedError()

    def __request(self, method: str, endpoint: str, params: Dict, sign: bool):
        url = parse.urljoin(self._base_url, endpoint)
        headers, params, body = self.prepare(method, endpoint, params, sign)
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            proxies=self._proxies,
            **body
        )
        data = self.parse(response)
        return data

    def get(self, endpoint: str, params: Dict = None, sign: bool = False):
        return self.__request(method='GET', endpoint=endpoint, params=params, sign=sign)

    def post(self, endpoint: str, params: Dict = None, sign: bool = False):
        return self.__request(method='POST', endpoint=endpoint, params=params, sign=sign)

    def delete(self, endpoint: str, params: Dict = None, sign: bool = False):
        return self.__request(method='DELETE', endpoint=endpoint, params=params, sign=sign)
