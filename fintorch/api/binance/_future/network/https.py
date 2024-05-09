import hashlib
import hmac
import json
from datetime import datetime
from typing import Dict, Tuple
from urllib import parse

from requests import Response

from .error import BinanceFutureHttpsException
from ...._api.network import Https


class BinanceFutureHttps(Https):
    def __init__(self, api_key: str = None, secret_key: str = None, proxies: Dict[str, str] = None):
        super().__init__(base_url='https://fapi.binance.com', proxies=proxies)
        self.__api_key: str = api_key
        self.__secret_key: str = secret_key

    def sign(self, method: str, endpoint: str, params: Dict, request_time: int) -> str:
        to_sign = parse.urlencode(params)
        return hmac.new(self.__secret_key.encode(), msg=to_sign.encode(), digestmod=hashlib.sha256).hexdigest()

    def prepare(self, method: str, endpoint: str, params: Dict, sign: bool) -> Tuple[Dict, Dict, Dict]:
        request_time = int(datetime.now().timestamp() * 1000)
        headers = {"X-MBX-APIKEY": self.__api_key}
        params = params if params is not None else {}
        params["timestamp"] = request_time
        if sign:
            params["signature"] = self.sign(method=method, endpoint=endpoint, params=params, request_time=request_time)

        return headers, params, {}

    def parse(self, response: Response):
        if 200 != response.status_code:
            raise BinanceFutureHttpsException(response.text)
        elif 400 == response.status_code:
            response_map = json.loads(response.text)
            raise BinanceFutureHttpsException(f"Code: {response_map['code']}, Message: {response_map['msg']}")
        else:
            return json.loads(response.text)
