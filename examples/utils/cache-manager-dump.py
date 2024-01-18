import numpy as np

from fintorch.utils.cache_manager import CacheManager


def main():
    dtype = np.float16
    arrays = [np.ones(2 ** 10).astype(dtype=dtype) * i for i in range(2 ** 13)]
    file_names = [f"file {i}" for i in range(len(arrays))]

    print(file_names)

    with CacheManager(directory="./tmp") as fs:
        for array, file_name in zip(arrays, file_names):
            fs.dump(obj=array, file_name=file_name)


if __name__ == '__main__':
    main()
