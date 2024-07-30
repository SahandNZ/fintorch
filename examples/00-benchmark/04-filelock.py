import os
import time
import pickle
import marshal
import filelock
from datetime import datetime

from fintorch.utils.directory import create_directory
from fintorch.settings import FINTORCH_TEMP_DIR



def main():
    file_path = os.path.join(FINTORCH_TEMP_DIR, "tmp.txt")
    lock_path = os.path.join(FINTORCH_TEMP_DIR, "tmp.txt.lock")
    with filelock.FileLock(lock_path):
        with open(file_path, "w+") as file:
            print("Started!")
            file.write("Started!\n")
            file.write(f"{str(datetime.now())}\n")
            time.sleep(10)
            file.write(f"{str(datetime.now())}\n")
            file.write("Done!\n")
            print("Done!")
            
    

if __name__ == '__main__':
    main()
