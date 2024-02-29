import argparse
import json
import os
from typing import Any, Dict, List

from fintorch.enum import TimeFrame
from fintorch.setting import NUM_HIDDEN_LAYERS, INTERVAL, DROPOUT, DIM_SEQUENCE, DIM_FEATURE, BATCH_NORM, DIM_OUTPUT, \
    ACTIVATION_FN, CONFIG_DIR, LR, WEIGHT_DECAY


class DefaultNamespace(argparse.Namespace):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

        self.transform_kwargs = {
            "symbol": self.symbol,
            "time_frame": self.time_frame,
            "dim_sequence": self.dim_sequence
        }

        self.model_kwargs = {
            "dim_sequence": self.dim_sequence,
            "dim_feature": self.dim_feature,
            "dim_output": self.dim_output,
            "num_hidden_layers": self.num_hidden_layers,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "activation_fn": ACTIVATION_FN
        }

        # load other args from config file
        self.config_path = os.path.join(CONFIG_DIR, f"{self.config_name}.json")
        with open(self.config_path, "r") as file:
            config_dict = json.load(file)

        self.symbols: List[str] = config_dict["symbols"]
        self.time_frames: List[TimeFrame] = [TimeFrame(tf) for tf in config_dict["time-frames"]]


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        # datetime args
        self.add_argument("--stop-date", action="store", type=str, required=False, default="2024-01-01")
        self.add_argument("--start-date", action="store", type=str, required=False, default="2019-01-01")

        # process args
        self.add_argument("--config-name", action="store", type=str, required=False, default="btc-daily")
        self.add_argument("--max-workers", action="store", type=int, required=False, default=os.cpu_count())

        # market args
        self.add_argument("--symbol", action="store", type=str, required=False, default="BTC-USDT")
        self.add_argument("--time-frame", action="store", type=int, required=False, default=TimeFrame.DAY1)

        # deep learning args
        self.add_argument("--interval", action="store", type=int, required=False, default=INTERVAL)
        self.add_argument("--dim-sequence", action="store", type=int, required=False, default=DIM_SEQUENCE)
        self.add_argument("--dim-feature", action="store", type=int, required=False, default=DIM_FEATURE)
        self.add_argument("--dim-output", action="store", type=int, required=False, default=DIM_OUTPUT)

        self.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=NUM_HIDDEN_LAYERS)
        self.add_argument("--dropout", action="store", type=float, required=False, default=DROPOUT)
        self.add_argument("--batch-norm", action="store", type=bool, required=False, default=BATCH_NORM)

        self.add_argument("--epochs", action="store", type=int, required=False, default=10)
        self.add_argument("--batch-size", action="store", type=int, required=False, default=128)
        self.add_argument("--lr", action="store", type=float, required=False, default=LR)
        self.add_argument("--weight-decay", action="store", type=float, required=False, default=WEIGHT_DECAY)

        # backtest args
        self.add_argument("--initial-capital", action="store", type=int, required=False, default=1000)
        self.add_argument("--speed", action="store", type=int, required=False, default=10000)

        self.__namespace: argparse.Namespace = None

    @staticmethod
    def parse(args: List[str] = None) -> DefaultNamespace:
        namespace = DefaultArgumentParser().parse_args(args=args)
        return DefaultNamespace(**namespace.__dict__)
