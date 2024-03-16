from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    df = ONLINE_EXCHANGE.future.data.get_candles_dataframe(symbol=args.symbol, time_frame=args.time_frame)
    has_missing_value = (0 != (df.index.diff().dropna() - args.time_frame)).max()
    print("Has df missing value?", "Yes" if has_missing_value else "No")


if __name__ == '__main__':
    main()
