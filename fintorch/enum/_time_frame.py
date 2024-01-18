from enum import IntEnum


class TimeFrame(IntEnum):
    MIN1 = 60
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

    def __str__(self) -> str:
        minute = self.value // 60
        hour = minute // 60
        day = hour // 24
        week = day // 7
        if minute < 60:
            return f"{minute} Minute" + ("s" if 1 < minute else "")
        elif hour < 24:
            return f"{hour} Hour" + ("s" if 1 < hour else "")
        elif day < 7:
            return f"{day} Day" + ("s" if 1 < day else "")
        elif week < 4:
            return f"{week} Week" + ("s" if 1 < week else "")
        else:
            return "1 Month"
