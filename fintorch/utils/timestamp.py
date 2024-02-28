from datetime import datetime
from typing import List, Union
import math


def to_datetime(date: Union[str, datetime]) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date


def to_timestamp(date: Union[str, datetime]) -> int:
    return int(to_datetime(date=date).timestamp())


def create_timestamps(start_date: Union[str, datetime], stop_date: Union[str, datetime], interval: int) -> List[int]:
    start_timestamp = math.ceil(to_timestamp(date=start_date) / interval) * interval
    stop_timestamp = math.floor(to_timestamp(date=stop_date) / interval) * interval
    return list(range(start_timestamp, stop_timestamp, interval))


def create_timestamps_divisions(start_date: str, stop_date: str, interval: int, divisions_count: int) \
        -> List[List[int]]:
    timestamps = create_timestamps(start_date=start_date, stop_date=stop_date, interval=interval)

    # create divisions
    divisions = []
    step = math.ceil(len(timestamps) / divisions_count)
    for start_index in range(0, len(timestamps), step):
        stop_index = start_index + step
        division = timestamps[start_index: stop_index]
        divisions.append(division)

    return divisions
