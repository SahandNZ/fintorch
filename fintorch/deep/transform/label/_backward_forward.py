from abc import ABC
from typing import Dict, Any

from ._label_transform import LabelTransform
from ....enum import TimeFrame


class BackwardForwardLabelTransform(LabelTransform, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            description: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            backward: int,
            forward: int,
            classes: list[str],
            data_loader_kwargs: Dict[str, Any],
            optimizer_kwargs: Dict[str, Any],
            lr_scheduler_kwargs: Dict[str, Any],
            trainer_kwargs: Dict[str, Any],
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            description=description,
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=forward,
            look_back=backward,
            classes=classes,
            data_loader_kwargs=data_loader_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            trainer_kwargs=trainer_kwargs
        )

        self.__backward: int = backward
        self.__forward: int = forward

    @property
    def backward(self) -> int:
        return self.__backward

    @property
    def forward(self) -> int:
        return self.__forward
