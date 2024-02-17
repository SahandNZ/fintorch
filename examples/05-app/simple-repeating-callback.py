import argparse
import time
from datetime import datetime

from fintorch.app import Application, Context


def callback(context: Context, sleep: int):
    print("callback called at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    time.sleep(sleep)
    print("callback done   at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', action='store', type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    app = Application.from_config(path=args.config_path)
    app.job_queue.run_repeating(callback=callback, kwargs={"sleep": 5}, interval=10, when='open', misfire_grace_time=2)
    app.job_queue.run_repeating(callback=callback, kwargs={"sleep": 2}, interval=10, when='open', misfire_grace_time=2)
    app.start()


if __name__ == '__main__':
    main()
