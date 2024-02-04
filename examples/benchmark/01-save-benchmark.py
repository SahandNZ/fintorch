import marshal
import os
import pickle
import time

import numpy as np

from fintorch.utils.directory import create_directory


def main():
    dtype = np.float16
    array = np.ones(2 ** 25).astype(dtype=dtype)

    tmp_directory = "../tmp"
    create_directory(tmp_directory)

    # file paths
    file_name = "array"
    pickle_file_path = os.path.join(tmp_directory, f"{file_name}.pkl")
    marshal_file_path = os.path.join(tmp_directory, f"{file_name}.bin")
    numpy_file_path = os.path.join(tmp_directory, f"{file_name}.npy")

    # save store with pickle
    pickle_start_time = time.perf_counter()
    with open(pickle_file_path, "wb+") as file:
        pickle.dump(array, file)
    pickle_time_usage = time.perf_counter() - pickle_start_time

    # save with marshal
    marshal_start_time = time.perf_counter()
    with open(marshal_file_path, "wb+") as file:
        marshal.dump(array.tobytes(), file)
    marshal_time_usage = time.perf_counter() - marshal_start_time

    # save with numpy
    numpy_start_time = time.perf_counter()
    with open(numpy_file_path, "wb+") as file:
        np.save(file, array)
    numpy_time_usage = time.perf_counter() - numpy_start_time

    # benchmarks
    pickle_disk_usage = os.path.getsize(pickle_file_path)
    marshal_disk_usage = os.path.getsize(marshal_file_path)
    numpy_disk_usage = os.path.getsize(numpy_file_path)

    print("{:^16} {:^32} {:^32}".format("Method", "Disk Usage (bytes)", "Time Usage (seconds)"))
    print("{:^16} {:^32} {:^32}".format("-" * 14, "-" * 30, "-" * 30))
    print("{:^16} {:^32} {:^32.3f}".format("Pickle.dump", pickle_disk_usage, pickle_time_usage))
    print("{:^16} {:^32} {:^32.3f}".format("Marshal.dump", marshal_disk_usage, marshal_time_usage))
    print("{:^16} {:^32} {:^32.3f}".format("Numpy.save", numpy_disk_usage, numpy_time_usage))


if __name__ == '__main__':
    main()
