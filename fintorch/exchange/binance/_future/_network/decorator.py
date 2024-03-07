from typing import Callable

from fintorch.utils.decorator import encode_param, decode_field
from .decoder import *
from .encoder import *


def encode_symbol(func: Callable):
    return encode_param(func=func, param='symbol', encoder=symbol_encoder)


def encode_time_frame(func: Callable):
    return encode_param(func=func, param='time_frame', encoder=time_frame_encoder)


def encode_order_side(func: Callable):
    return encode_param(func=func, param='side', encoder=order_side_encoder)


def encode_order_type(func: Callable):
    return encode_param(func=func, param='type', encoder=order_type_encoder)


def decode_symbol(func: Callable):
    return decode_field(func=func, field="symbol", decoder=symbol_decoder)


def decode_order_side(func: Callable):
    return decode_field(func=func, field="side", decoder=order_side_decoder)


def decode_order_type(func: Callable):
    return decode_field(func=func, field="type", decoder=order_type_decoder)


def decode_order_status(func: Callable):
    return decode_field(func=func, field='status', decoder=order_status_decoder)


def decode_position_side(func: Callable):
    return decode_field(func=func, field='side', decoder=position_side_decoder)
