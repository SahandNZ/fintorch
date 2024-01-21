from datetime import datetime
from typing import Dict, List

from ..metrics import Metrics


class Fold:
    def __init__(self, train_timestamps: List[int], validation_timestamps: List[int], test_timestamps: List[int]):
        self.__train_timestamps: List[int] = train_timestamps
        self.__validation_timestamps: List[int] = validation_timestamps
        self.__test_timestamp: List[int] = test_timestamps

        self.__epoch_to_train_metrics: Dict[int, Metrics] = {}
        self.__epoch_to_validation_metrics: Dict[int, Metrics] = {}
        self.__epoch_to_test_metrics: Dict[int, Metrics] = {}
        self.__epoch_to_model_state_dict: Dict[int, Dict] = {}

    @property
    def train_timestamps(self) -> List[int]:
        return self.__train_timestamps

    @property
    def validation_timestamps(self) -> List[int]:
        return self.__validation_timestamps

    @property
    def test_timestamps(self) -> List[int]:
        return self.__test_timestamp

    @property
    def train_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.train_timestamps[0])

    @property
    def validation_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.validation_timestamps[0])

    @property
    def test_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[0])

    @property
    def test_stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[-1])

    @property
    def epoch_to_train_metrics(self) -> Dict[int, Metrics]:
        return self.__epoch_to_train_metrics

    @property
    def epoch_to_validation_metrics(self) -> Dict[int, Metrics]:
        return self.__epoch_to_validation_metrics

    @property
    def epoch_to_test_metrics(self) -> Dict[int, Metrics]:
        return self.__epoch_to_test_metrics

    @property
    def epoch_to_model_state_dict(self) -> Dict[int, Dict]:
        return self.__epoch_to_model_state_dict

    @property
    def best_train_metrics(self) -> Metrics:
        return Metrics.get_best_metric(metrics_list=list(self.epoch_to_train_metrics.values()))

    @property
    def best_validation_metrics(self) -> Metrics:
        return Metrics.get_best_metric(metrics_list=list(self.epoch_to_validation_metrics.values()))

    @property
    def best_test_metrics(self) -> Metrics:
        return Metrics.get_best_metric(metrics_list=list(self.epoch_to_test_metrics.values()))
