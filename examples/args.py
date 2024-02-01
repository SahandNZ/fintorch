import argparse
import os

from fintorch.enum import TimeFrame


def add_default_args_and_parse(parser: argparse):
    # datetime args
    parser.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
    parser.add_argument("--start-date", action="store", type=str, required=False, default="2021-01-01")

    # boolean args
    parser.add_argument("--no-batch-norm", action="store_false")

    # integer args
    parser.add_argument("--epochs", action="store", type=int, required=False, default=20)
    parser.add_argument("--batch-size", action="store", type=int, required=False, default=1024)
    parser.add_argument("--dim-sequence", action="store", type=int, required=False, default=32)
    parser.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=2)
    parser.add_argument("--interval", action="store", type=int, required=False, default=TimeFrame.MIN15)
    parser.add_argument("--time-frame", action="store", type=int, required=False, default=TimeFrame.MIN15)
    parser.add_argument("--max-workers", action="store", type=int, required=False, default=os.cpu_count())

    # float args
    parser.add_argument("--dropout", action="store", type=float, required=False, default=0.5)

    # string args
    parser.add_argument("--config-path", action="store", type=str, required=False, default="./../config.json")
    args = parser.parse_args()

    return args
