import psutil
import torch


def print_cuda_memory_status():
    print("CUDA:\n"
          "\tTotal:     {:.2f} MB\n"
          "\tAllocated: {:.2f} MB\n"
          "\tReserved:  {:.2f} MB\n"
          .format(torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
                  torch.cuda.memory_allocated(0) / 2 ** 20,
                  torch.cuda.memory_reserved(0) / 2 ** 20))


def print_cpu_memory_status():
    process = psutil.Process()
    print("CPU:\n"
          "\tTotal:     {:.2f} MB\n"
          "\tAllocated: {:.2f} MB\n"
          .format(psutil.virtual_memory().total / 2 ** 20,
                  process.memory_info().rss / 2 ** 20))


def print_memory_status():
    print_cpu_memory_status()
    if torch.cuda.is_available():
        print_cuda_memory_status()
