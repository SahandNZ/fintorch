import time
from datetime import datetime

import logging.config

from fintorch.app import Context
from fintorch.utils.args import DefaultArgumentParser


def callback(context: Context, sleep: int):
    print("callback called at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    time.sleep(sleep)
    print("callback done   at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def main():
    args = DefaultArgumentParser.parse()
    logging.config.dictConfig(args.logging_config_dict)

    app = args.app
    app.job_queue.run_repeating(callback=callback, kwargs={"sleep": 5}, interval=10, when='open', misfire_grace_time=2)
    app.job_queue.run_repeating(callback=callback, kwargs={"sleep": 2}, interval=10, when='open', misfire_grace_time=2)
    app.start()


if __name__ == '__main__':
    main()
