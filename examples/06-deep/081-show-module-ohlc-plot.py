from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(
        symbols=[args.module.symbol],
        time_frames=[args.module.time_frame]
    )

    # show ohlc plot
    with args.module:
        args.module.show_ohlc_plot(dc=dc, start_date=args.start_date, stop_date=args.stop_date)


if __name__ == '__main__':
    main()
