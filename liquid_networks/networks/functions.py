from typing import Callable, Literal

import torch as th
from torch.nn import functional as th_f

ReductionType = Literal["sum", "batchmean"]

LossFunctionType = Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]


def _reduce(out: th.Tensor, reduction: ReductionType) -> th.Tensor:
    if reduction == "sum":
        return th.sum(out)
    if reduction == "batchmean":
        return th.mean(th.sum(out, dim=1), dim=0)
    raise ValueError(f"Unknown reduction {reduction}")


def cross_entropy_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.cross_entropy(outputs, targets, reduction="none").mean(dim=1),
        reduction,
    )


def cross_entropy(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.cross_entropy(outputs, targets, reduction="none"), reduction
    )


def mse_loss(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.mse_loss(outputs, targets, reduction="none"),
        reduction,
    )


def mse_loss_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.mse_loss(outputs, targets, reduction="none").mean(dim=1),
        reduction,
    )


def kl_div(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.kl_div(outputs.log(), targets, reduction="none"),
        reduction,
    )


def kl_div_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        th_f.kl_div(outputs.log(), targets, reduction="none").mean(dim=1),
        reduction,
    )
