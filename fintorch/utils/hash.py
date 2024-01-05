from typing import List


def list_hash(values: List) -> int:
    total_hash = 1
    for value in values:
        if isinstance(value, str):
            total_hash *= hash(value)
        elif isinstance(value, int):
            total_hash *= value

        if 10 ** 8 < total_hash:
            total_hash = total_hash % 10 ** 8

    return total_hash
