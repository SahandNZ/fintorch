import random
from tenacity import retry
from datetime import datetime

from fintorch.utils.directory import create_directory
from fintorch.settings import FINTORCH_TEMP_DIR

@retry
def do_something_unreliable():
    if random.randint(0, 10) > 5:
        raise IOError("Broken sauce, everything is hosed!!!111one")
    else:
        return "Awesome sauce!"


def main():
    print(do_something_unreliable())
    
            
    

if __name__ == '__main__':
    main()
