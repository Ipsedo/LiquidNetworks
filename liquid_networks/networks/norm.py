# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class TimeLayerNorm(nn.LayerNorm):
    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class TimeNorm(nn.Module):
    def __init__(
        self, features: int, affine: bool = True, eps: float = 1e-5
    ) -> None:
        super().__init__()

        self.__affine = affine

        self.__weight = (
            nn.Parameter(th.ones(1, features, 1)) if affine else None
        )
        self.__bias = (
            nn.Parameter(th.zeros(1, features, 1)) if affine else None
        )

        self.__epsilon = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = (x - th.mean(x, dim=1, keepdim=True)) / (
            th.std(x, dim=1, keepdim=True) + self.__epsilon
        )

        if self.__affine:
            out = out * self.__weight + self.__bias

        return out
