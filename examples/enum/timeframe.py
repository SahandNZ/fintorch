from fintorch.enum import TimeFrame


def main():
    print(TimeFrame.MIN5)
    print(TimeFrame.MIN5.short_str())
    print(TimeFrame.MIN1 * 20)


if __name__ == "__main__":
    main()
