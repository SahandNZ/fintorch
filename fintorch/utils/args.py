import argparse
import json
import os
from typing import Any, Dict, List

from fintorch.enum import TimeFrame, MarketType
from fintorch.setting import (
    CONFIG_DIR,

    SYMBOL,
    TIME_FRAME,
    EXCHANGE_NAME,

    INTERVAL,
    DIM_SEQUENCE,
    DIM_FEATURE,
    DIM_OUTPUT,
    NUM_HIDDEN_LAYERS,
    DROPOUT,
    BATCH_NORM,
    ACTIVATION_FN,

    LR,
    SHUFFLE,
    BATCH_SIZE,
    VAL_LENGTH,
    TEST_LENGTH,
    TRAIN_LENGTH,
    EPOCHS_COUNT,
    WEIGHT_DECAY,
    GRADIENT_CLIPPING_THRESHOLD,
)


class DefaultNamespace(argparse.Namespace):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)

        # load app kwargs from its config file
        path = os.path.join(CONFIG_DIR, f"{self.app_config}.json")
        with open(path, "r") as file:
            app_config_dict = json.load(file)

        # load symbols and time frame from their config file
        path = os.path.join(CONFIG_DIR, f"{self.stf_config}.json")
        with open(path, "r") as file:
            stf_config_dict = json.load(file)

        self.symbols: List[str] = stf_config_dict["symbols"]
        self.time_frames: List[TimeFrame] = [TimeFrame(tf) for tf in stf_config_dict["time-frames"]]

        # transform and model and app kwargs
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
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "activation_fn": ACTIVATION_FN
        }

        self.app_kwargs = app_config_dict
        self.app_kwargs.update({
            "symbols": self.symbols,
            "time_frames": self.time_frames,
            "interval": self.interval,
            "market_type": MarketType.FUTURE,
        })


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        # datetime args
        self.add_argument("--stop-date", action="store", type=str, required=False, default="2024-03-01")
        self.add_argument("--start-date", action="store", type=str, required=False, default="2020-01-01")

        # process args
        self.add_argument("--app-config", action="store", type=str, required=False, default="app")
        self.add_argument("--stf-config", action="store", type=str, required=False, default="btc-15m")
        self.add_argument("--max-workers", action="store", type=int, required=False, default=os.cpu_count())

        # market args
        self.add_argument("--symbol", action="store", type=str, required=False, default=SYMBOL)
        self.add_argument("--time-frame", action="store", type=int, required=False, default=TIME_FRAME)
        self.add_argument("--exchange-name", action="store", type=str, required=False, default=EXCHANGE_NAME)

        # deep learning args
        self.add_argument("--interval", action="store", type=int, required=False, default=INTERVAL)
        self.add_argument("--dim-sequence", action="store", type=int, required=False, default=DIM_SEQUENCE)
        self.add_argument("--dim-feature", action="store", type=int, required=False, default=DIM_FEATURE)
        self.add_argument("--dim-output", action="store", type=int, required=False, default=DIM_OUTPUT)

        self.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=NUM_HIDDEN_LAYERS)
        self.add_argument("--dropout", action="store", type=float, required=False, default=DROPOUT)
        self.add_argument("--batch-norm", action="store", type=bool, required=False, default=BATCH_NORM)

        self.add_argument("--lr", action="store", type=float, required=False, default=LR)
        self.add_argument("--shuffle", action="store", type=bool, required=False, default=SHUFFLE)
        self.add_argument("--batch-size", action="store", type=int, required=False, default=BATCH_SIZE)
        self.add_argument("--val-length", action="store", type=int, required=False, default=VAL_LENGTH)
        self.add_argument("--test-length", action="store", type=int, required=False, default=TEST_LENGTH)
        self.add_argument("--train-length", action="store", type=int, required=False, default=TRAIN_LENGTH)
        self.add_argument("--epochs-count", action="store", type=int, required=False, default=EPOCHS_COUNT)
        self.add_argument("--weight-decay", action="store", type=float, required=False, default=WEIGHT_DECAY)
        self.add_argument("--gct", action="store", type=float, required=False, default=GRADIENT_CLIPPING_THRESHOLD)

        # backtest args
        self.add_argument("--initial-capital", action="store", type=int, required=False, default=1000)
        self.add_argument("--speed", action="store", type=int, required=False, default=10000)

    @staticmethod
    def parse(args: List[str] = None) -> DefaultNamespace:
        namespace = DefaultArgumentParser().parse_args(args=args)
        return DefaultNamespace(**namespace.__dict__)
