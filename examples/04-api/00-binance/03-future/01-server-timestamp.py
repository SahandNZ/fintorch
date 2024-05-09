from datetime import datetime

from fintorch.utils.args import DefaultArgumentParser


def main():
    args = DefaultArgumentParser.parse()

    server_timestamp = args.api.future.data.get_server_timestamp()
    print(datetime.fromtimestamp(server_timestamp))
    print(server_timestamp)


if __name__ == '__main__':
    main()
