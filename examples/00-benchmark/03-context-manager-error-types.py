import marshal
import os
import pickle
import time

import numpy as np

from fintorch.utils.directory import create_directory


class EnterExitTest:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, (KeyboardInterrupt, Exception)):
            print("Some thing bad happened!!")


def main():
    with EnterExitTest():
        time.sleep(10)
        print(100 / 0)


if __name__ == '__main__':
    main()
