from ....enum import OrderSide, OrderStatus, OrderType, PositionSide


def symbol_decoder(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")


def order_side_decoder(order_side: str) -> int:
    return OrderSide.BUY if "BUY" == order_side else OrderSide.SELL


def order_type_decoder(order_type: str) -> int:
    return OrderType.LIMIT if "LIMIT" == order_type else OrderType.MARKET


def order_status_decoder(order_status: str) -> int:
    if 'NEW' == order_status:
        return OrderStatus.OPEN
    elif 'PARTIALLY_FILLED' == order_status:
        return OrderStatus.PARTIALLY_FILLED
    elif 'FILLED' == order_status:
        return OrderStatus.FILLED
    else:
        return OrderStatus.CANCELED


def position_side_decoder(position_side: str) -> int:
    return PositionSide.LONG if 'LONG' == position_side else PositionSide.SHORT
