# -*- coding: utf-8 -*-
import torch as th
from torch.nn import functional as F


def cross_entropy_time_series(
    outputs: th.Tensor, targets: th.Tensor
) -> th.Tensor:
    return (
        F.cross_entropy(outputs, targets, reduction="none").sum(dim=1).mean()
    )


def cross_entropy(outputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
    return F.cross_entropy(outputs, targets, reduction="mean")


def mse_loss(outputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
    return F.mse_loss(outputs, targets, reduction="none").sum(dim=-1).mean()


def kl_div(outputs: th.Tensor, targets: th.Tensor) -> th.Tensor:
    return F.kl_div(outputs, targets, reduction="batchmean")
