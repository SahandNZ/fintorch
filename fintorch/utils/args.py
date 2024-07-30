import os
import json
import argparse
import itertools
import logging.config
from typing import List, Dict, Any, Type

import yaml

from .function import call_with_dict
from .timestamp import to_timestamp

from ..enum import TimeFrame, MarketType
from ..settings import CONFIG_DIR


class DefaultNamespace(argparse.Namespace):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        
        #####################################
        ####### LEVEL 0 DEFAULR ARGS ########
        #####################################
        if 0 <= self.level:
            # set default values of cli args
            from ..defaults.zero import START_DATE, STOP_DATE, API_CONFIG, LOGGER_CONFIG, SYMBOLS_CONFIG, TIME_FRAMES
            
            self.start_date = self.start_date or START_DATE
            self.stop_date = self.stop_date or STOP_DATE            
            self.api_config = self.api_config or API_CONFIG
            self.logger_config = self.logger_config or LOGGER_CONFIG
            self.symbols_config = self.symbols_config or SYMBOLS_CONFIG
            
            # load and setup logger from its config file
            path = os.path.join(CONFIG_DIR, "logger", f"{self.logger_config}.yaml")
            with open(path, 'r') as f:
                self.logging_config_dict = yaml.safe_load(f.read())
            logging.config.dictConfig(self.logging_config_dict)

            # load api kwargs from its config file
            path = os.path.join(CONFIG_DIR, "api", f"{self.api_config}.json")
            with open(path, "r") as file:
                self.api_kwargs = json.load(file)

            # load symbols and time frame from their config file
            path = os.path.join(CONFIG_DIR, "symbols.json")
            with open(path, "r") as file:
                symbols = json.load(file)

            # set start and stop timestamps
            self.start_timestamp: float = to_timestamp(self.start_date)
            self.stop_timestamp: float = to_timestamp(self.stop_date)

            # set symbols and time frames
            self.symbols: List[str] = symbols[:self.symbols_config]
            self.time_frames: List[TimeFrame] = TIME_FRAMES

            self.symbol: str = self.symbols[0]
            self.time_frame: TimeFrame = self.time_frames[0]

        
        #####################################
        ##### LEVEL 1 MARKET DATA ARGS ######
        #####################################
        if 1 <= self.level:
            # set default values of cli args
            from ..defaults.one import MAX_TASKS, MAX_WORKERS, EXECUTOR_TYPE
            
            self.max_tasks = self.max_tasks or MAX_TASKS
            self.max_workers = self.max_workers or MAX_WORKERS
            self.executor_type = self.executor_type or EXECUTOR_TYPE
            
            import multiprocessing as mp
            from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
            from ..api.binance import BinanceAPI
            from ..app import Application
            from ..exchange import OnlineExchange
            
            # add default kwargs and instances of api and online exchange
            self.proxies = self.api_kwargs["proxies"]
            self.api_key = self.api_kwargs["api-key"]
            self.secret_key = self.api_kwargs["secret-key"]

            self.api = call_with_dict(BinanceAPI, self.api_kwargs)
            self.online_exchange = OnlineExchange(api=self.api)
            
            # add deafult kwargs and instance of application
            if "process" == self.executor_type.lower():
                mp_context = mp.get_context("spawn")
                # pool_kwargs = {"max_tasks_per_child": self.max_tasks, "mp_context": mp_context}
                pool_kwargs = {"mp_context": mp_context}
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers, pool_kwargs=pool_kwargs)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            self.app_kwargs = {
                "symbols": self.symbols,
                "time_frames": self.time_frames,
                "online_exchange": self.online_exchange,
                "market_type": MarketType.FUTURE,
                "executor": self.executor
            }
            self.app = Application(**self.app_kwargs)
            
        
        #####################################
        ######## LEVEL 2 DEEP ARGS ##########
        ##################################### 
        if 2 <= self.level:
            # set default values of cli args
            from ..defaults.two import DROPOUT, BATCH_NORM, DIM_INPUT_SEQUENCE, DIM_LATENT_SEQUENCE, \
                DIM_LATENT_FEATURE, DIM_OUTPUT_SEQUENCE, NUM_HIDDEN_LAYERS, \
                TRAIN_PERIOD, VAL_PERIOD, TEST_PERIOD, \
                BATCH_SIZE, BATCH_COUNT, \
                OPTIM_LR, OPTIM_WEIGHT_DECAY, \
                LRS_STEP_SIZE, LRS_GAMMA, \
                EPOCHS_COUNT, GRADIENT_CLIPPING_THRESHOLD
                
            self.dropout = self.dropout or DROPOUT
            self.batch_norm = self.batch_norm or BATCH_NORM
            self.num_hidden_layers = self.num_hidden_layers or NUM_HIDDEN_LAYERS
            
            self.dim_input_sequence = self.dim_input_sequence or DIM_INPUT_SEQUENCE
            self.dim_latent_sequence = self.dim_latent_sequence or DIM_LATENT_SEQUENCE
            self.dim_latent_feature = self.dim_latent_feature or DIM_LATENT_FEATURE
            self.dim_output_sequence = self.dim_output_sequence or DIM_OUTPUT_SEQUENCE
            
            self.train_period = self.train_period or TRAIN_PERIOD
            self.val_period = self.val_period or VAL_PERIOD
            self.test_period = self.test_period or TEST_PERIOD
            
            self.batch_size = self.batch_size or BATCH_SIZE
            self.batch_count = self.batch_count or BATCH_COUNT
            
            self.lr = self.lr or OPTIM_LR
            self.weight_decay = self.weight_decay or OPTIM_WEIGHT_DECAY
            
            self.lrs_step_size = self.lrs_step_size or LRS_STEP_SIZE
            self.lrs_gamma = self.lrs_gamma or LRS_GAMMA
            
            self.epochs_count = self.epochs_count or EPOCHS_COUNT
            self.gct = self.gct or GRADIENT_CLIPPING_THRESHOLD
            
                
            from ..deep.cross_validation import CrossValidation
            from ..deep.data_loader import DataLoader
            from ..deep.model import Model
            from ..deep.module import Module
            from ..deep.transform import Transform
            from ..deep.transform.feature import FeatureTransform
            from ..deep.transform.label import LabelTransform
            from ..defaults.two import FEATURE_TIME_FRAMES, LABEL_TIME_FRAMES, FEATURE_TRANSFORM_TYPES, \
                LABEL_TRANSFORM_TYPES, DATASET_TYPE, MODEL_TYPES, ACTIVATION_FN, OPTIM_TYPE, LRS_TYPE, CRITERION
            
            self.feature_time_frames: List[TimeFrame] = FEATURE_TIME_FRAMES
            self.label_time_frames: List[TimeFrame] = LABEL_TIME_FRAMES
            
            # transform types and kwargs
            self.feature_transform_types: List[Type[FeatureTransform]] = FEATURE_TRANSFORM_TYPES
            self.label_transform_types: List[Type[LabelTransform]] = LABEL_TRANSFORM_TYPES
            self.transform_types: List[Type[Transform]] = self.feature_transform_types + self.label_transform_types

            self.feature_transform_type = FEATURE_TRANSFORM_TYPES[0]
            self.label_transform_type = LABEL_TRANSFORM_TYPES[0]

            self.feature_transform_kwargs = {"dim_sequence": self.dim_input_sequence}
            self.label_transform_kwargs = {"dim_sequence": self.dim_output_sequence}

            # feature transform instances
            symbols = self.symbols
            time_frames = self.feature_time_frames
            transform_types = self.feature_transform_types

            self.feature_transforms = []
            for symbol, time_frame, type_ in itertools.product(symbols, time_frames, transform_types):
                kwargs = self.feature_transform_kwargs.copy()
                kwargs.update({"symbol": symbol, "time_frame": time_frame})
                feature_transform = call_with_dict(type_, kwargs)
                self.feature_transforms.append(feature_transform)

            self.feature_transform = self.feature_transforms[0]

            # label transform instances
            symbols = self.symbols
            time_frames = self.label_time_frames
            transform_types = self.label_transform_types

            self.label_transforms = []
            for symbol, time_frame, type_ in itertools.product(symbols, time_frames, transform_types):
                kwargs = self.label_transform_kwargs.copy()
                kwargs.update({"symbol": symbol, "time_frame": time_frame})
                label_transform = call_with_dict(type_, kwargs)
                self.label_transforms.append(label_transform)

            self.label_transform = self.label_transforms[0]

            # transform instances
            self.transforms = self.feature_transforms + self.label_transforms
            self.transform = self.transforms[0]

            # dataset type, kwargs, and instance
            self.dataset_type = DATASET_TYPE
            self.dataset_kwargs = {
                "feature_symbol": self.symbol,
                "feature_time_frame": self.time_frame,
                "feature_time_frames": self.feature_time_frames,
                "feature_transform_kwargs": self.feature_transform_kwargs,
                "feature_transform_type": self.feature_transform_type,

                "label_symbol": self.symbol,
                "label_time_frame": self.time_frame,
                "label_transform_kwargs": self.label_transform_kwargs,
                "label_transform_type": self.label_transform_type
            }
            self.dataset = call_with_dict(self.dataset_type, self.dataset_kwargs)

            # cross validation kwargs, and instance
            self.cross_validation_kwargs = {
                "train_period": self.train_period,
                "val_period": self.val_period,
                "test_period": self.test_period,
            }
            self.cross_validation = CrossValidation(time_frame=self.dataset.time_frame, **self.cross_validation_kwargs)

            # data loader, kwargs and instance
            self.data_loader_kwargs = {"batch_count": self.batch_count}
            self.data_loader = DataLoader(**self.data_loader_kwargs)

            # model types, kwargs, and instances
            self.model_types: List[Type[Model]] = MODEL_TYPES
            self.model_type = self.model_types[0]

            self.model_kwargs = {
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "activation_fn": ACTIVATION_FN,

                "dim_input_time_frame": self.dataset.dim_input_time_frame,
                "dim_input_sequence": self.dataset.dim_input_sequence,
                "dim_input_feature": self.dataset.dim_input_feature,

                "dim_latent_sequence": self.dim_latent_sequence,
                "dim_latent_feature": self.dim_latent_feature,

                "dim_output_time_frame": self.dataset.dim_output_time_frame,
                "dim_output_sequence": self.label_transform.dim_sequence,
                "dim_output_feature": self.label_transform.dim_feature,

                "num_hidden_layers": self.num_hidden_layers,
            }

            self.models = [call_with_dict(model_type, self.model_kwargs) for model_type in self.model_types]
            self.model = self.models[0]

            # module kwargs, and instances
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

            items = itertools.product(
                self.symbols,
                self.label_time_frames,
                self.feature_transform_types,
                self.label_transform_types,
                self.model_types
            )

            self.modules: List[Module] = []
            for symbol, time_frame, ft_type, lt_type, model_type in items:
                dataset_kwargs = self.dataset_kwargs.copy()
                dataset_kwargs.update({
                    "feature_symbol": symbol,
                    "feature_time_frames": self.feature_time_frames,
                    "feature_transform_type": ft_type,

                    "label_symbol": symbol,
                    "label_time_frame": time_frame,
                    "label_transform_type": lt_type,
                })
                dataset = call_with_dict(self.dataset_type, dataset_kwargs)

                module = Module(
                    dataset=dataset,
                    model_type=model_type,
                    model_kwargs=self.model_kwargs,
                    cross_validation_kwargs=self.cross_validation_kwargs,
                    data_loader_kwargs=self.data_loader_kwargs,
                    optimizer_kwargs=self.optimizer_kwargs,
                    lr_scheduler_kwargs=self.lr_scheduler_kwargs,
                    trainer_kwargs=self.trainer_kwargs
                )
                self.modules.append(module)

            self.module: Module = self.modules[0]

        #####################################
        ###### LEVEL 3 STRATEGY ARGS ########
        #####################################
        if 3 <= self.level:
            from ..defaults.three import INITIAL_CAPITAL, INTERVAL
            self.initial_capital = self.initial_capital or INITIAL_CAPITAL
            self.interval = self.interval or INTERVAL
            
            from ..engine import SimulationEngine
            from ..strategy import Strategy
            from ..defaults.three import STRATEGY_TYPES, STRATEGY_TIME_FRAMES
             
            # strategy types, kwargs, and instances
            self.strategy_types: List[Type[Strategy]] = STRATEGY_TYPES
            self.strategy_type: Type[Strategy] = self.strategy_types[0]

            self.strategy_kwargs = {"module": self.module}
            self.strategies = []
            for strategy_type, module in itertools.product(self.strategy_types, self.modules):
                if module.time_frame in STRATEGY_TIME_FRAMES and module.symbol in self.symbols:
                    strategy = strategy_type(module=module)
                    self.strategies.append(strategy)

            self.strategy: Strategy = self.strategies[0]

            # engine instances
            self.engines: List[SimulationEngine] = []
            for strategy in self.strategies:
                engine = SimulationEngine(
                    online_exchange=self.online_exchange,
                    strategy=strategy,
                    interval=self.interval
                )
                self.engines.append(engine)

            self.engine: SimulationEngine = self.engines[0]

class DefaultArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument("--level", action="store", type=int, required=False, default=3)
        
        #####################################
        ####### LEVEL 0 DEFAULR ARGS ########
        #####################################
        # datetime args
        self.add_argument("--start-date", action="store", type=str, required=False)
        self.add_argument("--stop-date", action="store", type=str, required=False)

        # config args
        self.add_argument("--api-config", action="store", type=str, required=False)
        self.add_argument("--logger-config", action="store", type=str, required=False)
        self.add_argument("--symbols-config", action="store", type=int, required=False)
        
        #####################################
        ##### LEVEL 1 MARKET DATA ARGS ######
        #####################################
        # application args
        self.add_argument("--max-tasks", action="store", type=int, required=False)
        self.add_argument("--max-workers", action="store", type=int, required=False)
        self.add_argument("--executor-type", action="store", type=str, required=False)

        #####################################
        ######## LEVEL 2 DEEP ARGS ##########
        ##################################### 
        self.add_argument("--dropout", action="store", type=float, required=False)
        self.add_argument("--batch-norm", action="store", type=bool, required=False)
        self.add_argument("--num-hidden-layers", action="store", type=int, required=False)

        self.add_argument("--dim-input-sequence", action="store", type=int, required=False)
        self.add_argument("--dim-latent-sequence", action="store", type=int, required=False)
        self.add_argument("--dim-latent-feature", action="store", type=int, required=False)
        self.add_argument("--dim-output-sequence", action="store", type=int, required=False)

        # cross validation args
        self.add_argument("--train-period", action="store", type=int, required=False)
        self.add_argument("--val-period", action="store", type=int, required=False)
        self.add_argument("--test-period", action="store", type=int, required=False)

        # data loader args
        self.add_argument("--batch-size", action="store", type=int, required=False)
        self.add_argument("--batch-count", action="store", type=int, required=False)

        # optimizer args
        self.add_argument("--lr", action="store", type=float, required=False)
        self.add_argument("--weight-decay", action="store", type=float, required=False)

        # lr scheduler args
        self.add_argument("--lrs-step-size", action="store", type=int, required=False)
        self.add_argument("--lrs-gamma", action="store", type=float, required=False)

        # trainer args
        self.add_argument("--epochs-count", action="store", type=int, required=False)
        self.add_argument("--no-shuffle", action="store_false", required=False)
        self.add_argument("--no-cuda", action="store_false", required=False)
        self.add_argument("--half-precision", action="store_true", required=False)
        self.add_argument("--gct", action="store", type=float, required=False)

        # backtest args
        self.add_argument("--initial-capital", action="store", type=int, required=False)
        self.add_argument("--interval", action="store", type=int, required=False)

        

    @staticmethod
    def parse(args: List[str] = None) -> DefaultNamespace:
        namespace = DefaultArgumentParser().parse_args(args=args)
        return DefaultNamespace(**namespace.__dict__)
