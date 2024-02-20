import argparse

from examples.args import add_default_args_and_parse
from fintorch.deep.transform.label import *
from fintorch.enum import TimeFrame
from fintorch.exchange import ONLINE_EXCHANGE


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # draw ohlc plot
    transform = NextFractalLabelTransform(symbol=args.symbol, time_frame=TimeFrame(args.time_frame))
    transform.show_ohlc_plot(dc=dc, start_date=args.start_date, stop_date=args.stop_date)


if __name__ == '__main__':
    main()
