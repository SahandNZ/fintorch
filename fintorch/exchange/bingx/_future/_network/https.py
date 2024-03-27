import hashlib
import hmac
import json
from typing import Dict, Tuple
from urllib import parse

from requests import Response

from ...._online import Https
from .exception import BingxFutureHttpsException


class BingxFutureHttps(Https):
    def __init__(self, key: str = None, secret_key: str = None, proxies: Dict[str, str] = None):
        super().__init__(base_url='https://open-api.bingx.com', key=key, secret_key=secret_key, proxies=proxies)

    def sign(self, method: str, endpoint: str, params: Dict, request_time: int) -> str:
        to_sign = parse.urlencode(params)
        return hmac.new(self._secret_key.encode(), msg=to_sign.encode(), digestmod=hashlib.sha256).hexdigest()

    def prepare(self, method: str, endpoint: str, params: Dict, sign: bool) -> Tuple[Dict, Dict, Dict]:
        header = {}
        params = {}
        body = {}

        return header, params, body

    def parse(self, response: Response):
        if 200 != response.status_code:
            raise BingxFutureHttpsException(response.text)
        elif 400 == response.status_code:
            response_map = json.loads(response.text)
            raise BingxFutureHttpsException(f"Code: {response_map['code']}, Message: {response_map['msg']}")
        else:
            return json.loads(response.text)["data"]
