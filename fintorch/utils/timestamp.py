import math
from datetime import datetime
from typing import List, Union

from fintorch.enum import TimeFrame


def to_datetime(date: Union[str, datetime]) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date


def to_timestamp(date: Union[str, datetime]) -> int:
    return int(to_datetime(date=date).timestamp())


def floor_timestamp(timestamp: int, time_frame: TimeFrame) -> int:
    return math.floor(timestamp / int(time_frame)) * int(time_frame)


def ceil_timestamp(timestamp: int, time_frame: TimeFrame) -> int:
    return math.ceil(timestamp / int(time_frame)) * int(time_frame)


def create_timestamps(
        start_date: Union[str, datetime],
        stop_date: Union[str, datetime],
        time_frame: TimeFrame
) -> List[int]:
    start_timestamp = floor_timestamp(timestamp=to_timestamp(date=start_date), time_frame=time_frame)
    stop_timestamp = ceil_timestamp(timestamp=to_timestamp(date=stop_date), time_frame=time_frame)
    return list(range(start_timestamp, stop_timestamp, int(time_frame)))


def create_timestamp_chunks(
        start_date: Union[str, datetime],
        stop_date: Union[str, datetime],
        time_frame: TimeFrame,
        divisions_count: int
) -> List[List[int]]:
    timestamps = create_timestamps(start_date=start_date, stop_date=stop_date, time_frame=time_frame)

    # create divisions
    divisions = []
    step = math.ceil(len(timestamps) / divisions_count)
    for start_index in range(0, len(timestamps), step):
        stop_index = start_index + step
        division = timestamps[start_index: stop_index]
        divisions.append(division)

    return divisions
