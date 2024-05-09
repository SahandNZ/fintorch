from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()
    ping = args.api.future.data.get_ping()
    print(ping)


if __name__ == '__main__':
    main()
