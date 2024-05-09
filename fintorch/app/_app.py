from typing import List

from ._context import Context
from ._job_queue import JobQueue
from ..enum import MarketType, TimeFrame
from ..exchange import OnlineExchange


class Application:
    def __init__(
            self,
            symbols: List[str],
            time_frames: List[TimeFrame],
            online_exchange: OnlineExchange,
            market_type: MarketType,
    ):
        self.__context = Context(
            symbols=symbols,
            time_frames=time_frames,
            online_exchange=online_exchange,
            market_type=market_type,
        )
        self.__job_queue = JobQueue(context=self.context)

    @property
    def context(self) -> Context:
        return self.__context

    @property
    def job_queue(self) -> JobQueue:
        return self.__job_queue

    def start(self, refresh_context: bool = True):
        if refresh_context:
            self.context.refresh()
        self.job_queue.start()
