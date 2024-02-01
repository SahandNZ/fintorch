from datetime import datetime
from typing import Dict, List, Tuple

from ..criterion import Criterion


class Fold:
    def __init__(self, index: int, train_timestamps: List[int], val_timestamps: List[int], test_timestamps: List[int]):
        self.__index: int = index
        self.__train_timestamps: List[int] = train_timestamps
        self.__val_timestamps: List[int] = val_timestamps
        self.__test_timestamp: List[int] = test_timestamps

        self.__criterion: Criterion = None
        self.__epoch_to_train_loss: Dict[int, float] = {}
        self.__epoch_to_validation_loss: Dict[int, float] = {}
        self.__epoch_to_test_loss: Dict[int, float] = {}
        self.__epoch_to_model_state_dict: Dict[int, Dict] = {}

    @property
    def index(self) -> int:
        return self.__index

    @property
    def start_date(self) -> datetime:
        return self.train_start_datetime

    @property
    def stop_date(self) -> datetime:
        return self.test_stop_datetime

    @property
    def train_timestamps(self) -> List[int]:
        return self.__train_timestamps

    @property
    def val_timestamps(self) -> List[int]:
        return self.__val_timestamps

    @property
    def test_timestamps(self) -> List[int]:
        return self.__test_timestamp

    @property
    def train_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.train_timestamps[0])

    @property
    def val_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.val_timestamps[0])

    @property
    def test_start_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[0])

    @property
    def test_stop_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.test_timestamps[-1])

    @property
    def criterion(self) -> Criterion:
        return self.__criterion

    @criterion.setter
    def criterion(self, value: Criterion) -> None:
        self.__criterion = value

    @property
    def epoch_to_train_loss(self) -> Dict[int, float]:
        return self.__epoch_to_train_loss

    @property
    def epoch_to_val_loss(self) -> Dict[int, float]:
        return self.__epoch_to_validation_loss

    @property
    def epoch_to_test_loss(self) -> Dict[int, float]:
        return self.__epoch_to_test_loss

    @property
    def epoch_to_model_state_dict(self) -> Dict[int, Dict]:
        return self.__epoch_to_model_state_dict

    @property
    def best_train_epoch(self) -> int:
        return self.__best_epoch_and_loss(self.epoch_to_train_loss)

    @property
    def best_val_epoch(self) -> int:
        return self.__best_epoch_and_loss(self.epoch_to_val_loss)

    @property
    def best_test_epoch(self) -> int:
        return self.__best_epoch_and_loss(self.epoch_to_test_loss)

    @property
    def best_train_loss(self) -> float:
        return self.epoch_to_train_loss[self.best_train_epoch]

    @property
    def best_val_loss(self) -> float:
        return self.epoch_to_val_loss[self.best_val_epoch]

    @property
    def best_test_loss(self) -> float:
        return self.epoch_to_test_loss[self.best_test_epoch]

    @property
    def best_val_on_test_loss(self) -> float:
        return self.epoch_to_test_loss[self.best_val_epoch]

    @property
    def best_train_model_state_dict(self) -> Dict:
        return self.epoch_to_model_state_dict[self.best_train_epoch]

    @property
    def best_val_model_state_dict(self) -> Dict:
        return self.epoch_to_model_state_dict[self.best_val_epoch]

    @property
    def best_test_model_state_dict(self) -> Dict:
        return self.epoch_to_model_state_dict[self.best_test_epoch]

    def __best_epoch_and_loss(self, fold_to_loss: Dict[int, float]) -> int:
        best_epoch, best_loss = None, None
        for epoch, loss in fold_to_loss.items():
            if best_loss is None or self.criterion.compare(best_loss, loss):
                best_loss = loss
                best_epoch = epoch

        return best_epoch
