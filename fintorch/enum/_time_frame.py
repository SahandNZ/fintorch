from enum import Enum


class TimeFrame(float, Enum):
    MS50 = 0.05
    MS100 = 2 * MS50
    MS250 = 5 * MS50
    MS500 = 10 * MS50
    MS750 = 15 * MS50

    SEC1 = 20 * MS50
    SEC5 = 5 * SEC1
    SEC10 = 10 * SEC1
    SEC15 = 15 * SEC1
    SEC20 = 20 * SEC1
    SEC30 = 30 * SEC1
    SEC45 = 45 * SEC1

    MIN1 = 60 * SEC1
    MIN5 = 5 * MIN1
    MIN15 = 15 * MIN1
    MIN30 = 30 * MIN1

    HOUR1 = 60 * MIN1
    HOUR2 = 2 * HOUR1
    HOUR4 = 4 * HOUR1
    HOUR6 = 6 * HOUR1
    HOUR8 = 8 * HOUR1
    HOUR12 = 12 * HOUR1

    DAY1 = 24 * HOUR1
    DAY3 = 3 * DAY1
    WEEK1 = 7 * DAY1
    MONTH1 = 30 * DAY1

    @staticmethod
    def upper(value: int):
        last_index = None
        for index, time_frame in enumerate(TimeFrame):
            if time_frame <= value:
                last_index = index

        next_index = min(last_index + 1, len(TimeFrame) - 1)
        return list(TimeFrame)[next_index]

    def __str__(self) -> str:
        second = self.value
        minute = second // 60
        hour = minute // 60
        day = hour // 24
        week = day // 7

        if second < 0:
            return f"{int(self.value * 1000)} Milli-Second" + ("s" if 1 < self.value else "")
        elif second < 60:
            return f"{int(second)} Second" + ("s" if 1 < second else "")
        elif minute < 60:
            return f"{int(minute)} Minute" + ("s" if 1 < minute else "")
        elif hour < 24:
            return f"{int(hour)} Hour" + ("s" if 1 < hour else "")
        elif day < 7:
            return f"{int(day)} Day" + ("s" if 1 < day else "")
        elif week < 4:
            return f"{int(week)} Week" + ("s" if 1 < week else "")
        else:
            return "1 Month"
