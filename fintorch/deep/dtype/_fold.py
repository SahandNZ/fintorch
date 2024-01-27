from datetime import datetime
from typing import Dict, List, Tuple

from ..criterion import Criterion


class Fold:
    def __init__(self, train_timestamps: List[int], validation_timestamps: List[int], test_timestamps: List[int]):
        self.__train_timestamps: List[int] = train_timestamps
        self.__validation_timestamps: List[int] = validation_timestamps
        self.__test_timestamp: List[int] = test_timestamps

        self.__criterion: Criterion = None
        self.__epoch_to_train_loss: Dict[int, float] = {}
        self.__epoch_to_validation_loss: Dict[int, float] = {}
        self.__epoch_to_test_loss: Dict[int, float] = {}
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
    def criterion(self) -> Criterion:
        return self.__criterion

    @criterion.setter
    def criterion(self, value: Criterion) -> None:
        self.__criterion = value

    @property
    def epoch_to_train_loss(self) -> Dict[int, float]:
        return self.__epoch_to_train_loss

    @property
    def epoch_to_validation_loss(self) -> Dict[int, float]:
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
    def best_validation_epoch(self) -> int:
        return self.__best_epoch_and_loss(self.epoch_to_validation_loss)

    @property
    def best_test_epoch(self) -> int:
        return self.__best_epoch_and_loss(self.epoch_to_test_loss)

    @property
    def best_train_loss(self) -> float:
        return self.epoch_to_train_loss[self.best_train_epoch]

    @property
    def best_validation_loss(self) -> float:
        return self.epoch_to_validation_loss[self.best_validation_epoch]

    @property
    def best_test_loss(self) -> float:
        return self.epoch_to_test_loss[self.best_test_epoch]

    @property
    def best_train_model_state_dict(self) -> Dict:
        return self.epoch_to_model_state_dict[self.best_train_epoch]

    @property
    def best_validation_model_state_dict(self) -> Dict:
        return self.epoch_to_model_state_dict[self.best_validation_epoch]

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
