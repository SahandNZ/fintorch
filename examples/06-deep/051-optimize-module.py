from rich.live import Live
from rich.panel import Panel

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    dc = args.online_exchange.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # optimize folds
    with Live(refresh_per_second=2) as live, args.module:
        for status in args.module.optimize(dc=dc, start_date=args.start_date, stop_date=args.stop_date):
            live.update(Panel.fit(str(status), title=f"{str(args.module)}"))


if __name__ == '__main__':
    main()
