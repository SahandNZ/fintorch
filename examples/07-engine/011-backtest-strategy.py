import logging.config

from fintorch.utils.args import DefaultArgumentParser
from fintorch.strategy import ActiveMarketTrailingStopLossStrategy

def main():
    string_args = ["--symbols-config", "10"]
    args = DefaultArgumentParser.parse(args=string_args)
    logging.config.dictConfig(args.logging_config_dict)
    
    symbol = args.symbols[9]
    print(symbol)
    engines = [
        e for e in args.engines 
        if symbol == e.strategy.symbol 
        and isinstance(e.strategy, ActiveMarketTrailingStopLossStrategy)
    ]
    engine = engines[0]
    print(len(engines))
    print(engine)
    
    with engine:
        for _ in engine.simulate():
            pass

        measures = engine.calculate_measures(start_date=args.start_date, stop_date=args.stop_date)
        print("\n", str(engine.strategy), "\n", measures)


if __name__ == '__main__':
    main()
