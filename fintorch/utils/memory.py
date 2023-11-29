import sys
from typing import Dict

import psutil
import torch


def get_cpu_memory_status(start: str = "", end: str = "\n") -> str:
    process = psutil.Process()
    return "{}CPU:{}" \
           "{}\tTotal:     {:.2f} MB{}" \
           "{}\tAllocated: {:.2f} MB{}" \
        .format(start, end,
                start, psutil.virtual_memory().total / 2 ** 20, end,
                start, process.memory_info().rss / 2 ** 20, end)


def get_cuda_memory_status(start: str = "", end: str = "\n") -> str:
    return "{}CUDA:{}" \
           "{}\tTotal:     {:.2f} MB{}" \
           "{}\tAllocated: {:.2f} MB{}" \
           "{}\tReserved:  {:.2f} MB{}" \
        .format(start, end,
                start, torch.cuda.get_device_properties(0).total_memory / 2 ** 20, end,
                start, torch.cuda.memory_allocated(0) / 2 ** 20, end,
                start, torch.cuda.memory_reserved(0) / 2 ** 20, end)


def get_memory_status(start: str = "", end: str = "\n") -> str:
    result = "{}Memory status:{}".format(start, end)
    sub_start = start + "\t"
    if torch.cuda.is_available():
        result += get_cpu_memory_status(start=sub_start, end=end) + get_cuda_memory_status(start=sub_start, end=end)
    else:
        result += get_cpu_memory_status(start=sub_start, end=end)

    return result


def format_size(size_: int) -> str:
    for unit in ["", "K", "M", "G"]:
        if size_ < 2 ** 10:
            return "{:.2f} {}B".format(size_, unit)
        size_ /= 2 ** 10

    return "{:.2f} TB".format(size_)


def get_local_variables_memory_usage(locals_: Dict, count: int = None) -> str:
    sorted_locals = sorted(locals_.items(), key=lambda item: sys.getsizeof(item[1]), reverse=True)
    if count is not None:
        sorted_locals = sorted_locals[:count]
    result = ""
    for name, value in sorted_locals:
        result += "{:<64}{}\n".format(name, format_size(sys.getsizeof(value)))

    return result
