def symbol_encoder(symbol: str) -> str:
    return symbol.replace('-', '')


def time_frame_encoder(time_frame: int) -> int:
    minute = time_frame // 60
    hour = minute // 60
    day = hour // 24
    week = day // 7
    if minute < 60:
        return "{}m".format(minute)
    elif hour < 24:
        return "{}h".format(hour)
    elif day < 7:
        return "{}d".format(day)
    elif week < 4:
        return "{}w".format(week)
    else:
        return "1M"


def order_side_encoder(order_side: int) -> str:
    return "BUY" if 1 == order_side else "SELL"


def order_type_encoder(order_type: int) -> str:
    return "LIMIT" if 1 == order_type else "MARKET"
