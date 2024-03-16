from datetime import datetime
from typing import Dict

from .....dtype import Candle, Order, SymbolInfo, Position


def candle_deserializer(data: Dict):
    instance = Candle()

    instance.timestamp = int(int(data[0]) / 1000)
    instance.open = float(data[1])
    instance.high = float(data[2])
    instance.low = float(data[3])
    instance.close = float(data[4])
    instance.volume = float(data[5])
    instance.trade = int(data[8])

    return instance


def candle_wss_deserializer(data: Dict):
    instance = Candle()

    instance.timestamp = data["t"] / 1000
    instance.open = float(data["o"])
    instance.high = float(data["h"])
    instance.low = float(data["l"])
    instance.close = float(data["c"])
    instance.volume = float(data["v"])
    instance.trade = int(data["n"])

    return instance


def order_deserializer(data: Dict):
    instance = Order()

    instance.id = data['orderId']
    instance.symbol = data['symbol']
    instance.datetime = datetime.fromtimestamp(int(data['time']) / 1000)
    instance.timestamp = int(int(data['time']) / 1000)
    instance.type = data['type']
    instance.status = data['status']
    instance.side = data['side']
    instance.volume = float(data['origQty'])
    instance.price = float(data['price'])

    return instance


def position_deserializer(data: Dict):
    instance = Position()

    instance.symbol = data['symbol']
    instance.type = data['marginType']
    instance.volume = abs(float(data['positionAmt']))
    instance.entry_price = float(data['entryPrice'])
    instance.leverage = float(data['leverage'])
    instance.liquidation_price = float(data['liquidationPrice'])

    return instance


def symbol_info_deserializer(data: Dict):
    instance = SymbolInfo()

    instance.symbol = data['symbol']
    instance.base_asset = data['baseAsset']
    instance.quote_asset = data['quoteAsset']
    instance.on_board_timestamp = int(data['onboardDate']) // 1000
    instance.price_precision = int(data['pricePrecision'])
    instance.quantity_precision = int(data['quantityPrecision'])

    return instance
