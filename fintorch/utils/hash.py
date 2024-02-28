from typing import List, Union

def static_hash(value: Union[str, int, float]) -> int:
    if isinstance(value, str):
        hash_value = int.from_bytes(value.encode(), byteorder="big") % (17 ** 8)
    elif isinstance(value, int) or isinstance(value, float):
        hash_value = value % (11 ** 8)
    elif isinstance(value, bool):
        hash_value = int(value)
    else:
        hash_value = 1

    return hash_value


def static_list_hash(values: List) -> int:
    hash_value = 1
    for value in values:
        hash_value = (hash_value * static_hash(value)) % (13 ** 8)

    return int(hash_value)
