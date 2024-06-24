import logging.config

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    logging.config.dictConfig(args.logging_config_dict)

    with args.engine:
        for _ in args.engine.simulate():
            pass

        measures = args.engine.calculate_measures(start_date=args.start_date, stop_date=args.stop_date)
        print("\n", str(args.engine.strategy), "\n", measures)


if __name__ == '__main__':
    main()
