from typing import List


def list_hash(values: List) -> int:
    total_hash = 1
    for value in values:
        if isinstance(value, str):
            total_hash = (total_hash * hash(value)) % 10 ** 8
        if isinstance(value, int):
            total_hash = (total_hash * value) % 10 ** 8

    return total_hash
