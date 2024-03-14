# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.__kernel_size = 3
        super().__init__(
            in_channels,
            out_channels,
            (self.__kernel_size,),
            padding=(0,),
            stride=(1,),
        )

        self.__padding = self.__kernel_size // 2 + self.__kernel_size % 2

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(F.pad(x, (self.__padding, 0)))[
            :, :, : -self.__padding
        ]
