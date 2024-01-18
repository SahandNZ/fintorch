from typing import List

from .decorator import *
from ..._exchange import Https, Wss, Trade
from ....dtype import Balance, Order, Position


class BinanceFutureTrade(Trade):
    def __init__(self, https: Https, wss: Wss):
        super().__init__(https, wss)

    def get_balance(self) -> Balance:
        endpoint = "/fapi/v2/balance"
        response = self._https.get(endpoint=endpoint, sign=True)
        response = [item for item in response if 'USDT' == item['asset']][0]
        balance = Balance.from_binance(response)
        return balance

    def get_leverage(self, symbol: str) -> int:
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol}
        response = self._https.get(endpoint=endpoint, params=params, sign=True)
        leverage = response[0]['leverage']
        return leverage

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        endpoint = "/fapi/v1/leverage"
        params = {"symbol": symbol, "leverage": leverage}
        response = self._https.post(endpoint=endpoint, params=params, sign=True)
        leverage = response['leverage']
        return leverage

    def get_order(self, symbol: str, order_id: str) -> Order:
        endpoint = "/fapi/v1/openOrders"
        params = {"symbol": symbol, "orderId": order_id}
        response = self._https.get(endpoint=endpoint, params=params, sign=True)
        orders = Order.from_binance(response)
        return orders

    def get_open_orders(self, symbol: str) -> List[Order]:
        endpoint = "/fapi/v1/openOrders"
        params = {"symbol": symbol}
        response = self._https.get(endpoint=endpoint, params=params, sign=True)
        orders = [Order.from_binance(item) for item in response]
        return orders

    def set_market_order(self, symbol: str, side: OrderSide, volume: float) -> str:
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": volume}
        response = self._https.post(endpoint=endpoint, params=params, sign=True)
        order_id = response["orderId"]
        return order_id

    def set_limit_order(self, symbol: str, side: OrderSide, volume: float, price: float) -> str:
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol, "side": side, "type": "LIMIT", "quantity": volume, "price": price,
                  "timeInForce": "GTC"}
        response = self._https.post(endpoint=endpoint, params=params, sign=True)
        order_id = response["orderId"]
        return order_id

    def set_stop_market_order(self, symbol: str, side: OrderSide, volume: float, stop_price: float) -> str:
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol, "side": side, "type": "LIMIT", "quantity": volume, "stopPrice": stop_price}
        response = self._https.post(endpoint=endpoint, params=params, sign=True)
        order_id = response["orderId"]
        return order_id

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol, "orderId": order_id}
        response = self._https.delete(endpoint=endpoint, params=params, sign=True)
        return True

    def cancel_all_orders(self, symbol: str) -> bool:
        endpoint = "/fapi/v1/allOpenOrders"
        params = {"symbol": symbol}
        response = self._https.delete(endpoint=endpoint, params=params, sign=True)
        return True

    def get_open_position(self, symbol: str) -> Position:
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol}
        response = self._https.get(endpoint=endpoint, params=params, sign=True)
        position = Position.from_binance(response[0])
        return position
