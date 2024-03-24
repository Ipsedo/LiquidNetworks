# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        kernel_size = 3
        stride = 1

        super().__init__(
            in_channels,
            out_channels,
            (kernel_size,),
            padding=(0,),
            stride=(stride,),
            dilation=dilation,
        )

        self.__padding = (kernel_size - 1) * dilation

        def __init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=1e-3)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-3)

        self.apply(__init_weights)

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(F.pad(x, (self.__padding, 0)))
