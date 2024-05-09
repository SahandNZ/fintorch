import argparse
import itertools
import json
import logging.config

import yaml

from .function import call_with_dict
from .timestamp import to_timestamp
from ..api.binance import BinanceAPI
from ..app import Application
from ..deep.cross_validation import CrossValidation
from ..deep.data_loader import DataLoader
from ..deep.dtype import Dataset
from ..deep.module import Module
from ..deep.transform import Transform
from ..deep.transform.label import LabelTransform
from ..defaults import *
from ..engine import SimulationEngine
from ..enum import TimeFrame, MarketType
from ..exchange import OnlineExchange
from ..settings import CONFIG_DIR
from ..strategy.deep import DeepStrategy


class DefaultNamespace(argparse.Namespace):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        # load and setup logger from its config file
        path = os.path.join(CONFIG_DIR, "logger", f"{self.logger_config}.yaml")
        with open(path, 'r') as f:
            self.logging_config_dict = yaml.safe_load(f.read())
        logging.config.dictConfig(self.logging_config_dict)

        # load api kwargs from its config file
        path = os.path.join(CONFIG_DIR, f"{self.api_config}.json")
        with open(path, "r") as file:
            self.api_kwargs = json.load(file)

        # load symbols and time frame from their config file
        path = os.path.join(CONFIG_DIR, "stf", f"{self.stf_config}.json")
        with open(path, "r") as file:
            stf_config_dict = json.load(file)

        # set start and stop timestamps
        self.start_timestamp: float = to_timestamp(self.start_date)
        self.stop_timestamp: float = to_timestamp(self.stop_date)

        # set symbols and time frames
        self.symbol: str = stf_config_dict["symbols"][0]
        self.time_frame: TimeFrame = TimeFrame(max(stf_config_dict["time-frames"]))

        self.symbols: List[str] = stf_config_dict["symbols"]
        self.time_frames: List[TimeFrame] = [TimeFrame(tf) for tf in stf_config_dict["time-frames"]]

        # add default kwargs and instances of api and exchange
        self.proxies = self.api_kwargs["proxies"]
        self.api_key = self.api_kwargs["api-key"]
        self.secret_key = self.api_kwargs["secret-key"]

        self.api = call_with_dict(BinanceAPI, self.api_kwargs)
        self.online_exchange = OnlineExchange(api=self.api)

        # deep types
        self.feature_transform_type = FEATURE_TRANSFORM_TYPES[0]
        self.label_transform_type = LABEL_TRANSFORM_TYPES[0]
        self.model_type = MODEL_TYPES[0]

        self.feature_transform_types: List[Type[FeatureTransform]] = FEATURE_TRANSFORM_TYPES
        self.label_transform_types: List[Type[LabelTransform]] = LABEL_TRANSFORM_TYPES
        self.transform_types: List[Type[Transform]] = TRANSFORM_TYPES
        self.model_types: List[Type[Model]] = MODEL_TYPES

        # deep kwargs
        self.transform_kwargs = {
            "symbol": self.symbol,
            "time_frame": self.time_frame,
            "dim_input_sequence": self.dim_input_sequence,
            "dim_output_sequence": self.dim_output_sequence
        }

        self.model_kwargs = {
            "num_hidden_layers": self.num_hidden_layers,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "activation_fn": ACTIVATION_FN
        }

        self.cross_validation_kwargs = {
            "train_period": self.train_period,
            "val_period": self.val_period,
            "test_period": self.test_period,
        }

        self.data_loader_kwargs = {"batch_size": self.batch_size}

        self.optimizer_kwargs = {
            "torch_optimizer_type": OPTIM_TYPE,
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }

        self.lr_scheduler_kwargs = {
            "torch_lr_scheduler_type": LRS_TYPE,
            "step_size": self.lrs_step_size,
            "gamma": self.lrs_gamma
        }

        self.trainer_kwargs = {
            "criterion": CRITERION,
            "epochs_count": self.epochs_count,
            "shuffle": self.no_shuffle,
            "auto_cuda": self.no_cuda,
            "half_precision": self.half_precision,
            "gradient_clipping_threshold": self.gct,
        }

        # deep instances
        self.feature_transforms = []
        for s, tf, ft_type in itertools.product(self.symbols, self.time_frames, self.feature_transform_types):
            ft_kwargs = self.transform_kwargs.copy()
            ft_kwargs.update({"dim_sequence": self.dim_input_sequence, "symbol": s, "time_frame": tf})
            feature_transform = call_with_dict(ft_type, ft_kwargs)
            self.feature_transforms.append(feature_transform)

        self.label_transforms = []
        for s, tf, lt_type in itertools.product(self.symbols, self.time_frames, self.label_transform_types):
            lt_kwargs = self.transform_kwargs.copy()
            lt_kwargs.update({"dim_sequence": self.dim_output_sequence, "symbol": s, "time_frame": tf})
            label_transform = call_with_dict(lt_type, lt_kwargs)
            self.label_transforms.append(label_transform)

        self.transforms = self.feature_transforms + self.label_transforms

        # create modules
        items = itertools.product(self.symbols, self.time_frames, self.label_transform_types,
                                  self.feature_transform_types, self.model_types)
        self.modules = []
        for s, tf, lt_type, ft_type, model_type in items:
            ft_kwargs = self.transform_kwargs.copy()
            ft_kwargs.update({"dim_sequence": self.dim_input_sequence, "symbol": s, "time_frame": tf})
            feature_transform = call_with_dict(ft_type, ft_kwargs)

            lt_kwargs = self.transform_kwargs.copy()
            lt_kwargs.update({"dim_sequence": self.dim_output_sequence, "symbol": s, "time_frame": tf})
            label_transform = call_with_dict(lt_type, lt_kwargs)

            module = Module(
                feature_transform=feature_transform,
                label_transform=label_transform,
                model_type=model_type,
                model_kwargs=self.model_kwargs,
                cross_validation_kwargs=self.cross_validation_kwargs,
                data_loader_kwargs=self.data_loader_kwargs,
                optimizer_kwargs=self.optimizer_kwargs,
                lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                trainer_kwargs=self.trainer_kwargs
            )
            self.modules.append(module)

        self.label_transform = self.label_transforms[0]
        self.feature_transform = self.feature_transforms[0]
        self.dataset = Dataset(feature_transform=self.feature_transform, label_transform=self.label_transform)
        self.cross_validation = CrossValidation(time_frame=self.dataset.time_frame, **self.cross_validation_kwargs)
        self.data_loader = DataLoader(**self.data_loader_kwargs)
        self.module = Module(
            feature_transform=self.feature_transform,
            label_transform=self.label_transform,
            model_type=self.model_type,
            model_kwargs=self.model_kwargs,
            cross_validation_kwargs=self.cross_validation_kwargs,
            data_loader_kwargs=self.data_loader_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
            trainer_kwargs=self.trainer_kwargs
        )

        model_kwargs = self.model_kwargs.copy()
        model_kwargs.update({
            "dim_input_sequence": self.feature_transform.dim_sequence,
            "dim_input_feature": self.feature_transform.dim_feature,
            "dim_output_sequence": self.label_transform.dim_sequence,
            "dim_output_feature": self.label_transform.dim_feature,
        })
        self.models = [call_with_dict(mt, model_kwargs) for mt in self.model_types]
        self.model = self.models[0]

        # add default deep strategy types and kwargs and instances
        self.deep_strategy_type: Type[DeepStrategy] = DEEP_STRATEGY_TYPES[0]
        self.deep_strategy_kwargs = {"module": self.module}
        self.deep_strategy = call_with_dict(self.deep_strategy_type, self.deep_strategy_kwargs)

        self.deep_strategy_types: List[Type[DeepStrategy]] = DEEP_STRATEGY_TYPES
        self.deep_strategies = []
        for deep_strategy_type, module in itertools.product(self.deep_strategy_types, self.modules):
            if TimeFrame.DAY1 == module.time_frame:
                deep_strategy = deep_strategy_type(module=module)
                self.deep_strategies.append(deep_strategy)

        # create engine instances
        self.engine: SimulationEngine = SimulationEngine(
            online_exchange=self.online_exchange,
            strategy=self.deep_strategy,
            interval=self.interval
        )

        self.engines: List[SimulationEngine] = []
        for strategy in self.deep_strategies:
            engine = SimulationEngine(
                online_exchange=self.online_exchange,
                strategy=strategy,
                interval=self.interval
            )
            self.engines.append(engine)

        # add app instance
        self.app_kwargs = {
            "symbols": self.symbols,
            "time_frames": self.time_frames,
            "online_exchange": self.online_exchange,
            "market_type": MarketType.FUTURE
        }
        self.app = Application(**self.app_kwargs)


class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        # datetime args
        self.add_argument("--start-date", action="store", type=str, required=False, default=START_DATE)
        self.add_argument("--stop-date", action="store", type=str, required=False, default=STOP_DATE)

        # config args
        self.add_argument("--api-config", action="store", type=str, required=False, default=API_CONFIG)
        self.add_argument("--stf-config", action="store", type=str, required=False, default=STF_CONFIG)
        self.add_argument("--logger-config", action="store", type=str, required=False, default=LOGGER_CONFIG)
        self.add_argument("--max-workers", action="store", type=int, required=False, default=MAX_WORKERS)

        # deep learning args
        self.add_argument("--dim-input-sequence", action="store", type=int, required=False, default=DIM_INPUT_SEQUENCE)
        self.add_argument("--dim-output-sequence", action="store", type=int, required=False,
                          default=DIM_OUTPUT_SEQUENCE)
        self.add_argument("--num-hidden-layers", action="store", type=int, required=False, default=NUM_HIDDEN_LAYERS)
        self.add_argument("--dropout", action="store", type=float, required=False, default=DROPOUT)
        self.add_argument("--batch-norm", action="store", type=bool, required=False, default=BATCH_NORM)

        # cross validation args
        self.add_argument("--train-period", action="store", type=int, required=False, default=TRAIN_PERIOD)
        self.add_argument("--val-period", action="store", type=int, required=False, default=VAL_PERIOD)
        self.add_argument("--test-period", action="store", type=int, required=False, default=TEST_PERIOD)

        # data loader args
        self.add_argument("--batch-size", action="store", type=int, required=False, default=BATCH_SIZE)

        # optimizer args
        self.add_argument("--lr", action="store", type=float, required=False, default=OPTIM_LR)
        self.add_argument("--weight-decay", action="store", type=float, required=False, default=OPTIM_WEIGHT_DECAY)

        # lr scheduler args
        self.add_argument("--lrs-step-size", action="store", type=float, required=False, default=LRS_STEP_SIZE)
        self.add_argument("--lrs-gamma", action="store", type=float, required=False, default=LRS_GAMMA)

        # trainer args
        self.add_argument("--epochs-count", action="store", type=int, required=False, default=EPOCHS_COUNT)
        self.add_argument("--no-shuffle", action="store_false", required=False)
        self.add_argument("--no-cuda", action="store_false", required=False)
        self.add_argument("--half-precision", action="store_true", required=False)
        self.add_argument("--gct", action="store", type=float, required=False, default=GRADIENT_CLIPPING_THRESHOLD)

        # backtest args
        self.add_argument("--initial-capital", action="store", type=int, required=False, default=INITIAL_CAPITAL)
        self.add_argument("--interval", action="store", type=int, required=False, default=INTERVAL)

    @staticmethod
    def parse(args: List[str] = None) -> DefaultNamespace:
        namespace = DefaultArgumentParser().parse_args(args=args)
        return DefaultNamespace(**namespace.__dict__)
