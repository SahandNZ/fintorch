import copy
from typing import Dict, List

from ._context import Context
from ._job_queue import JobQueue
from ..enum import MarketType, TimeFrame
from ..exchange import OnlineExchange
from ..utils.function import call_with_dict


class Application:
    def __init__(
            self,
            online_exchange: OnlineExchange,
            market_type: MarketType,
            symbols: List[str],
            time_frames: List[TimeFrame]
    ):
        self.__context = Context(
            online_exchange=online_exchange,
            market_type=market_type,
            symbols=symbols,
            time_frames=time_frames
        )
        self.__job_queue = JobQueue(context=self.context)

    @staticmethod
    def from_dict(dct: Dict):
        dct = copy.deepcopy(dct)
        dct['online-exchange'] = OnlineExchange.from_dict(dct)
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
