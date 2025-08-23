from typing import Callable, Literal

import torch as th
from torch.nn import functional as th_f

ReductionType = Literal["sum", "batchmean"]

LossFunctionType = Callable[[th.Tensor, th.Tensor, ReductionType], th.Tensor]


def _reduce(out: th.Tensor, reduction: ReductionType) -> th.Tensor:
    assert len(out.size()) == 1

    if reduction == "sum":
        return th.sum(out)
    if reduction == "batchmean":
        return th.mean(out)

    raise ValueError(f"Unknown reduction {reduction}")


def cross_entropy_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 3
    return _reduce(
        th_f.cross_entropy(outputs, targets, reduction="none").mean(dim=1),
        reduction,
    )


def cross_entropy(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 2
    return _reduce(
        th_f.cross_entropy(outputs, targets, reduction="none"),
        reduction,
    )


def soft_cross_entropy(
    proba: th.Tensor, target_proba: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(proba.size()) == 2
    return _reduce(-(target_proba * proba.log()).sum(dim=1), reduction)


def mse_loss(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 2
    return _reduce(
        th_f.mse_loss(outputs, targets, reduction="none").sum(dim=1),
        reduction,
    )


def mse_loss_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 3
    return _reduce(
        th_f.mse_loss(outputs, targets, reduction="none")
        .sum(dim=2)
        .mean(dim=1),
        reduction,
    )


def kl_div(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 2
    return _reduce(
        th_f.kl_div(outputs.log(), targets, reduction="none").sum(dim=1),
        reduction,
    )


def kl_div_time_series(
    outputs: th.Tensor, targets: th.Tensor, reduction: ReductionType
) -> th.Tensor:
    assert len(outputs.size()) == 3
    return _reduce(
        th_f.kl_div(outputs.log(), targets, reduction="none")
        .sum(dim=2)
        .mean(dim=1),
        reduction,
    )
