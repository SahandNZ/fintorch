from rich.progress import Progress

from fintorch.engine import SimulationEngine
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.setting import RICH_PROGRESS_COLUMNS
from fintorch.strategy.technical.sma import SmaStrategy
from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    # define strategy
    strategy = SmaStrategy(symbol=args.symbol, time_frame=args.time_frame)

    # define simulation engine
    engine = SimulationEngine(
        online_exchange=ONLINE_EXCHANGE,
        initial_capital=args.initial_capital,
        interval=args.interval,
        speed=args.speed,
    )

    with Progress(*RICH_PROGRESS_COLUMNS) as progress:
        engine.livetest(strategy=strategy, start_date=args.start_date, stop_date=args.stop_date, progress=progress)


if __name__ == '__main__':
    main()
