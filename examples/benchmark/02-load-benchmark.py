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

    # load with pickle
    pickle_start_time = time.perf_counter()
    with open(pickle_file_path, "rb") as file:
        pickle_array = pickle.load(file)
    pickle_time_usage = time.perf_counter() - pickle_start_time

    # load with marshal
    marshal_start_time = time.perf_counter()
    with open(marshal_file_path, "rb") as file:
        marshal_array = marshal.load(file)
    marshal_time_usage = time.perf_counter() - marshal_start_time

    # save with numpy
    numpy_start_time = time.perf_counter()
    with open(numpy_file_path, "rb") as file:
        np_array = np.load(file)
    numpy_time_usage = time.perf_counter() - numpy_start_time

    print("{:^16} {:^32}".format("Method", "Time Usage (seconds)"))
    print("{:^16} {:^32}".format("-" * 14, "-" * 30))
    print("{:^16} {:^32.3f}".format("Pickle.load", pickle_time_usage))
    print("{:^16} {:^32.3f}".format("Marshal.load", marshal_time_usage))
    print("{:^16} {:^32.3f}".format("Numpy.load", numpy_time_usage))


if __name__ == '__main__':
    main()
