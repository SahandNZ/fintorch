import argparse

from examples.args import add_default_args_and_parse
from fintorch.exchange import ONLINE_EXCHANGE


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    df = ONLINE_EXCHANGE.future.data.get_candles_dataframe(symbol=args.symbol, time_frame=args.time_frame)
    print((0 != (df.index.diff().dropna() - args.time_frame)).max())


if __name__ == '__main__':
    main()
