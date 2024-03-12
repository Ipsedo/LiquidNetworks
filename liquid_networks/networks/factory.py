# -*- coding: utf-8 -*-
from typing import Callable, Dict, Final, Literal, Type

import torch as th

from .functions import cross_entropy, cross_entropy_time_series, mse_loss
from .recurent import (
    LiquidRecurrent,
    LiquidRecurrentBrainActivity,
    LiquidRecurrentClf,
    LiquidRecurrentReg,
    LiquidRecurrentSingleClf,
)

_MODEL_DICT: Final[Dict[str, Type[LiquidRecurrent]]] = {
    "regression": LiquidRecurrentReg,
    "classification": LiquidRecurrentClf,
    "single_classification": LiquidRecurrentSingleClf,
    "brain_activity": LiquidRecurrentBrainActivity,
}

_LOSS_DICT: Final[Dict[str, Callable[[th.Tensor, th.Tensor], th.Tensor]]] = {
    "regression": mse_loss,
    "classification": cross_entropy_time_series,
    "single_classification": cross_entropy,
    "brain_activity": mse_loss,
}

TaskType = Literal[
    "regression", "classification", "single_classification", "brain_activity"
]


def get_model_constructor(task_type: TaskType) -> Type[LiquidRecurrent]:
    return _MODEL_DICT[task_type]


def get_loss_function(
    task_type: TaskType,
) -> Callable[[th.Tensor, th.Tensor], th.Tensor]:
    return _LOSS_DICT[task_type]
