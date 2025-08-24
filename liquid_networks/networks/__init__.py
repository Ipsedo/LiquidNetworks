from enum import StrEnum
from typing import Callable, Final

import torch as th
from torch.nn import functional as th_f

from .abstract_recurent import AbstractLiquidRecurrent, AbstractLiquidRecurrentFactory
from .losses import (
    LossFunctionType,
    cross_entropy,
    cross_entropy_time_series,
    kl_div,
    mse_loss_time_series,
)
from .recurrents import (
    BfrbLiquidRecurrentFactory,
    BrainActivityLiquidRecurrentFactory,
    LastLiquidRecurrentFactory,
    LiquidRecurrentFactory,
    SigmoidLiquidRecurrentFactory,
    SoftplusLiquidRecurrentFactory,
)

ModelFactoryConstructor = Callable[
    [dict[str, str]],
    AbstractLiquidRecurrentFactory,
]


class TaskType(StrEnum):
    REGRESSION = "regression"
    POSITIVE_REGRESSION = "positive_regression"
    CLASSIFICATION = "classification"
    MULTI_LABELS = "multi_labels"
    LAST_CLASSIFICATION = "last_classification"
    BRAIN_ACTIVITY = "brain_activity"
    BFRB = "bfrb"


class ActivationFunction(StrEnum):
    MISH = "mish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"


_MODEL_DICT: Final[dict[TaskType, ModelFactoryConstructor]] = {
    TaskType.REGRESSION: LiquidRecurrentFactory,
    TaskType.POSITIVE_REGRESSION: SoftplusLiquidRecurrentFactory,
    TaskType.CLASSIFICATION: LiquidRecurrentFactory,
    TaskType.MULTI_LABELS: SigmoidLiquidRecurrentFactory,
    TaskType.LAST_CLASSIFICATION: LastLiquidRecurrentFactory,
    TaskType.BRAIN_ACTIVITY: BrainActivityLiquidRecurrentFactory,
    TaskType.BFRB: BfrbLiquidRecurrentFactory,
}

_LOSS_DICT: Final[dict[TaskType, LossFunctionType]] = {
    TaskType.REGRESSION: mse_loss_time_series,
    TaskType.POSITIVE_REGRESSION: mse_loss_time_series,
    TaskType.CLASSIFICATION: cross_entropy_time_series,
    TaskType.MULTI_LABELS: mse_loss_time_series,
    TaskType.LAST_CLASSIFICATION: cross_entropy,
    TaskType.BRAIN_ACTIVITY: cross_entropy,
    TaskType.BFRB: cross_entropy,
}

_ACT_FN_DICT: Final[dict[ActivationFunction, Callable[[th.Tensor], th.Tensor]]] = {
    ActivationFunction.MISH: th_f.mish,
    ActivationFunction.RELU: th_f.relu,
    ActivationFunction.SIGMOID: th_f.sigmoid,
    ActivationFunction.TANH: th_f.tanh,
    ActivationFunction.LEAKY_RELU: th_f.leaky_relu,
    ActivationFunction.GELU: th_f.gelu,
}


def get_model_constructor(task_type: TaskType) -> ModelFactoryConstructor:
    return _MODEL_DICT[task_type]


def get_loss_function(task_type: TaskType) -> LossFunctionType:
    return _LOSS_DICT[task_type]


def get_activation_fn(act_fn: ActivationFunction) -> Callable[[th.Tensor], th.Tensor]:
    return _ACT_FN_DICT[act_fn]
