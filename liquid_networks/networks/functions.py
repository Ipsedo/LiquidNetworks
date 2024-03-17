# -*- coding: utf-8 -*-
from typing import Literal

import torch as th
from torch.nn import functional as F

ReductionType = Literal["sum", "batchmean"]


def _reduce(out: th.Tensor, reduction: ReductionType) -> th.Tensor:
    if reduction == "sum":
        return th.sum(out)
    if reduction == "batchmean":
        return th.mean(out)
    raise ValueError(f"Unknown reduction {reduction}")


def cross_entropy_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        F.cross_entropy(outputs, targets, reduction="none").sum(dim=1),
        reduction,
    )


def cross_entropy(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        F.cross_entropy(outputs, targets, reduction="none"), reduction
    )


def mse_loss(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        F.mse_loss(outputs, targets, reduction="none").sum(dim=-1), reduction
    )


def kl_div(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    return _reduce(
        F.kl_div(outputs.log(), targets, reduction="none").sum(dim=1),
        reduction,
    )
