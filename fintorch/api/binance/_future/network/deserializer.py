from typing import Dict

from .....dtype import Ticker, Candle, Order, SymbolInfo, Position, FundingRate, LongShortRatio, AggregatedTrade


def ticker_deserializer(data: Dict) -> Ticker:
    instance = Ticker()

    instance.symbol = data["symbol"]
    instance.timestamp = int(data["time"]) / 1000
    instance.price = float(data["price"])

    return instance


def aggregated_trade_deserializer(data: Dict) -> AggregatedTrade:
    instance = AggregatedTrade()

    instance.timestamp = int(data["T"]) / 1000
    instance.price = float(data["p"])
    instance.quantity = float(data["q"])
    instance.side = 1 if data["m"] else -1

    return instance


def candle_deserializer(data: Dict) -> Candle:
    instance = Candle()

    instance.timestamp = int(data[0]) / 1000
    instance.open = float(data[1])
    instance.high = float(data[2])
    instance.low = float(data[3])
    instance.close = float(data[4])
    instance.volume = float(data[5])
    instance.trade = int(data[8])

    return instance


def funding_rate_deserializer(data: Dict) -> FundingRate:
    instance = FundingRate()

    instance.timestamp = int(data["fundingTime"]) / 1000
    instance.price = float(data["markPrice"]) if 0 < len(data["markPrice"]) else None
    instance.rate = float(data["fundingRate"])

    return instance


def long_short_ratio_deserializer(data: Dict) -> LongShortRatio:
    instance = LongShortRatio()

    instance.timestamp = int(data["timestamp"]) / 1000
    instance.ratio = float(data["longShortRatio"])
    instance.long = float(data["longAccount"])
    instance.short = float(data["shortAccount"])

    return instance


def order_deserializer(data: Dict):
    instance = Order()

    instance.id = data['orderId']
    instance.symbol = data['symbol']
    instance.timestamp = int(data['time']) / 1000
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
    instance.on_board_timestamp = int(data['onboardDate']) / 1000
    instance.price_precision = int(data['pricePrecision'])
    instance.quantity_precision = int(data['quantityPrecision'])

    return instance
