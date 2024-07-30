from memory_profiler import profile
from fintorch.utils.args import DefaultArgumentParser


@profile
def main():
    string_args = [
        "--level", "3",
        "--symbols-config", "10",
        "--max-workers", "20"
    ]
    args = DefaultArgumentParser.parse(args=string_args)
    
            
if __name__ == '__main__':
    main()
