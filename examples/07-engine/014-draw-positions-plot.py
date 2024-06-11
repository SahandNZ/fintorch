import logging.config

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    logging.config.dictConfig(args.logging_config_dict)

    symbols = [args.engine.strategy.module.symbol]
    time_frames = args.engine.strategy.module.dataset.time_frames
    dc = args.online_exchange.future.data.get_data_collection(symbols=symbols, time_frames=time_frames)

    with args.engine:
        args.engine.show_positions_plot(dc=dc, start_date=args.start_date, stop_date=args.stop_date)


if __name__ == '__main__':
    main()
