from datetime import datetime

from fintorch.enum import TimeFrame
from fintorch.exchange import OnlineExchange
from fintorch.setting import PROXIES
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    current_open_timestamp = datetime.now().timestamp() // args.time_frame * args.time_frame

    exchange = OnlineExchange.from_name(name="binance", interval=TimeFrame.MIN1, proxies=PROXIES)
    exchange.future.data.prepare(symbols=[args.symbol], time_frames=[args.time_frame])
    exchange.future.data.next(timestamp=current_open_timestamp)

    # get server time
    server_timestamp = exchange.future.data.get_current_timestamp()
    print("{:<64}{}".format("Server timestamp", server_timestamp))
    print("{:<64}{}".format("Server datetime", datetime.fromtimestamp(server_timestamp)))

    # get ping in seconds
    ping = exchange.future.data.get_ping()
    print("{:<64}{}".format("Ping (seconds)", ping))

    # get symbols info
    symbols_info = exchange.future.data.get_symbols_info()
    first_symbol_info = symbols_info[0]
    print("{:<64}{}".format(f"Symbol info {first_symbol_info.symbol}", first_symbol_info.price_precision))

    # get latest tickers
    ticker = exchange.future.data.get_symbols_ticker(symbols=args.symbols)[0]
    print("{:<64}{}".format(f"Last ticker of {ticker.symbol} at {ticker.datetime}", ticker.price))
    
    # get last candles
    candle = exchange.future.data.get_current_candle(symbol=args.symbol, time_frame=args.time_frame)
    print("{:<64}{}".format(f"Last candle {candle.datetime}", candle.open))


if __name__ == '__main__':
    main()
