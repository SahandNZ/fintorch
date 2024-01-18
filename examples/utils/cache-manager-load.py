import numpy as np

from fintorch.utils.file_manager import FileManager


def main():
    file_name = "file 5"
    with FileManager(directory="./tmp") as fs:
        array = fs.load(file_name=file_name)

    print(array)


if __name__ == '__main__':
    main()
