from fintorch.enum import OrderSide, OrderStatus, OrderType, PositionSide


def symbol_decoder(symbol: str) -> str:
    return symbol.replace("USDT", "-USDT")


def order_side_decoder(order_side: str) -> int:
    return OrderSide.BUY if "BUY" == order_side else OrderSide.SELL


def order_type_decoder(order_type: str) -> OrderType:
    return OrderType.LIMIT if "LIMIT" == order_type else OrderType.MARKET


def order_status_decoder(order_status: str) -> OrderStatus:
    if 'NEW' == order_status:
        return OrderStatus.OPEN
    elif 'PARTIALLY_FILLED' == order_status:
        return OrderStatus.PARTIALLY_FILLED
    elif 'FILLED' == order_status:
        return OrderStatus.FILLED
    else:
        return OrderStatus.CANCELED


def position_side_decoder(position_side: str) -> PositionSide:
    return PositionSide.LONG if 'LONG' == position_side else PositionSide.SHORT
