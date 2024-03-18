# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        dilation = 1
        kernel_size = 3
        stride = 2

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size,),
            padding=(0,),
            stride=(stride,),
            dilation=dilation,
        )

        self.__padding = (kernel_size - 1) * dilation

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(F.pad(x, (self.__padding, 0)))
