from typing import Callable, Final, Literal

import torch as th
from torch.nn import functional as th_f

from .abstract_recurent import AbstractLiquidRecurrent
from .functions import (
    LossFunctionType,
    cross_entropy,
    cross_entropy_time_series,
    kl_div,
    mse_loss_time_series,
)
from .recurrents import (
    BfrbLiquidRecurrent,
    BrainActivityLiquidRecurrent,
    LastLiquidRecurrent,
    LiquidRecurrent,
    SigmoidLiquidRecurrent,
    SoftplusLiquidRecurrent,
)

_ModuleConstructorType = Callable[
    [int, int, int, Callable[[th.Tensor], th.Tensor], int],
    AbstractLiquidRecurrent,
]

_MODEL_DICT: Final[dict[str, _ModuleConstructorType]] = {
    "regression": LiquidRecurrent,
    "positive_regression": SoftplusLiquidRecurrent,
    "classification": LiquidRecurrent,
    "multi_labels": SigmoidLiquidRecurrent,
    "last_classification": LastLiquidRecurrent,
    "brain_activity": BrainActivityLiquidRecurrent,
    "bfrb": BfrbLiquidRecurrent,
}

_LOSS_DICT: Final[dict[str, LossFunctionType]] = {
    "regression": mse_loss_time_series,
    "positive_regression": mse_loss_time_series,
    "classification": cross_entropy_time_series,
    "multi_labels": mse_loss_time_series,
    "last_classification": cross_entropy,
    "brain_activity": kl_div,
    "bfrb": cross_entropy,
}

_ACT_FN_DICT: Final[dict[str, Callable[[th.Tensor], th.Tensor]]] = {
    "mish": th_f.mish,
    "relu": th_f.relu,
    "sigmoid": th_f.sigmoid,
    "tanh": th_f.tanh,
    "leaky_relu": th_f.leaky_relu,
    "gelu": th_f.gelu,
}

TaskType = Literal[
    "regression",
    "positive_regression",
    "classification",
    "multi_labels",
    "last_classification",
    "brain_activity",
    "bfrb",
]

ActivationFunction = Literal[
    "mish", "tanh", "sigmoid", "relu", "gelu", "leaky_relu"
]


def get_model_constructor(task_type: TaskType) -> _ModuleConstructorType:
    return _MODEL_DICT[task_type]


def get_loss_function(task_type: TaskType) -> LossFunctionType:
    return _LOSS_DICT[task_type]


def get_activation_fn(
    act_fn: ActivationFunction,
) -> Callable[[th.Tensor], th.Tensor]:
    return _ACT_FN_DICT[act_fn]
