from ._exchange import *
from ._local import *
from ._online import *
from ..setting import EXCHANGE_NAME, PROXIES, TRADING_INTERVAL

ONLINE_EXCHANGE: OnlineExchange = OnlineExchange.from_name(
    name=EXCHANGE_NAME,
    interval=TRADING_INTERVAL,
    proxies=PROXIES
)
