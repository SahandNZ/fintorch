from ._exchange import Exchange
from ._exchange import Future
from ._exchange import Market
from ._exchange import Spot
from ._exchange import Trade
from ._exchange import Wallet

from ..setting import EXCHANGE_NAME

EXCHANGE = Exchange.from_name(name=EXCHANGE_NAME)
del EXCHANGE_NAME
