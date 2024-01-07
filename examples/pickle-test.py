import marshal
import os

import numpy as np


def main():
    one_hot = np.zeros(2)
    one_hot[0] = 1
    print(one_hot)

    file_path = "./one_hot.pkl"
    with open(file_path, "wb+") as file:
        marshal.dump(one_hot, file)

    print(os.path.getsize(file_path))


if __name__ == '__main__':
    main()
