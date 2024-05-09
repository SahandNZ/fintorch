from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    agg_trades = args.api.future.data.get_aggregated_trades(
        symbol=args.symbol,
        start_timestamp=args.start_timestamp,
        stop_timestamp=args.stop_timestamp
    )


if __name__ == '__main__':
    main()
