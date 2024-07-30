from datetime import datetime

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    
    for symbol in args.symbols:
        df = args.online_exchange.future.data.get_base_candles_df(symbol=symbol, time_frame=args.time_frame)
        has_missing_value = (0 != (df.index.diff().dropna() - args.time_frame)).max()

        print(symbol)
        print("Has df missing value?", "Yes" if has_missing_value else "No")
        print(datetime.fromtimestamp(df.index[0]))
        print()


if __name__ == '__main__':
    main()
