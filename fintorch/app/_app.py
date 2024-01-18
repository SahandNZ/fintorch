import copy
import json
from typing import Dict, List

from ._context import Context
from ._job_queue import JobQueue
from ..enum import TimeFrame
from ..exchange import Exchange
from ..utils.function import call_with_dict


class Application:
    def __init__(self, exchange: Exchange, symbols: List[str], time_frames: List[TimeFrame]):
        self.__context = Context(exchange=exchange, symbols=symbols, time_frames=time_frames)
        self.__job_queue = JobQueue(context=self.context)

    @staticmethod
    def from_config(path: str):
        with open(path, 'r') as file:
            config_dct = json.load(file)

        exchange_dct = {"name": config_dct["exchange"]["name"]}
        app_dct = {"symbols": config_dct["symbols"], "time-frames": config_dct["time-frames"]}
        dct = {"exchange": exchange_dct, "app": app_dct}
        app = Application.from_dict(dct=dct)
        return app

    @staticmethod
    def from_dict(dct: Dict):
        exchange = Exchange.from_dict(dct["exchange"])

        dct = copy.deepcopy(dct["app"])
        dct['exchange'] = exchange
        app = call_with_dict(Application, dct)

        return app

    @property
    def context(self) -> Context:
        return self.__context

    @property
    def job_queue(self) -> JobQueue:
        return self.__job_queue

    def start(self):
        self.job_queue.start()
