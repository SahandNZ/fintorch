import os
import sys


def clear_console():
    os.system("clear")


def print_console(x: int, y: int, text: str):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()
