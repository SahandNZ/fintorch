from typing import Callable


def encode_param(func: Callable, param: str, encoder: Callable):
    def inner(*args, **kwargs):
        if encoder is not None and param in kwargs:
            kwargs[param] = encoder(kwargs[param])

        return func(*args, **kwargs)

    return inner


def decode_field(func: Callable, field: str, decoder: Callable):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, list):
            for item in result:
                if field in item.__dict__:
                    item.__dict__[field] = decoder(item.__dict__[field])
        elif field in result.__dict__:
            result.__dict__[field] = decoder(result.__dict__[field])

        return result

    return inner
