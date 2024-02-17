from fintorch.enum import OrderSide


def main():
    print(OrderSide.BUY)
    print(OrderSide.BUY.reverse)
    print(OrderSide.BUY * -1)


if __name__ == "__main__":
    main()
