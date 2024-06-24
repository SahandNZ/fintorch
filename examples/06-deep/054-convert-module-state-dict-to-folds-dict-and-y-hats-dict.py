from fintorch.utils.args import DefaultArgumentParser

def main():
    args = DefaultArgumentParser.parse()
    
    for module in args.modules[:1]:
        print(module.directory)


if __name__ == '__main__':
    main()
