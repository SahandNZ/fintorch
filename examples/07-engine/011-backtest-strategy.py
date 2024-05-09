import logging.config

from fintorch.engine import Measures
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    logging.config.dictConfig(args.logging_config_dict)

    with args.engine:
        for timestamp in args.engine.simulate():
            pass

    positions = args.engine.exchange.future.trade.get_positions_history(symbol=args.engine.strategy.symbol)
    measures = Measures(
        engine=args.engine,
        positions=positions,
        leverage=1,
        initial_capital=1000,
        margin_assignment_method="cumulative"
    )

    print(measures)


if __name__ == '__main__':
    main()
