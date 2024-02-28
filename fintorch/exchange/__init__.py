from ._exchange import *
from ._local import *
from ._online import *
from ..setting import EXCHANGE_NAME, PROXIES, INTERVAL

ONLINE_EXCHANGE: OnlineExchange = OnlineExchange.from_name(
    name=EXCHANGE_NAME,
    interval=INTERVAL,
    proxies=PROXIES
)
