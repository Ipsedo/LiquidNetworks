import math
from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as th_f

from .simple import SoftmaxLiquidRecurrent


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
        )

        self.__kernel_size = kernel_size

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(th_f.pad(x, (self.__kernel_size - 1, 0)))


class CausalConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            CausalConv1d(in_channels, out_channels, 3, 1),
            nn.Mish(),
            nn.InstanceNorm1d(out_channels),
            CausalConv1d(out_channels, out_channels, 3, 2),
            nn.Mish(),
            nn.InstanceNorm1d(out_channels),
        )


class BrainActivityLiquidRecurrent(SoftmaxLiquidRecurrent):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        output_size: int,
    ) -> None:
        factor = math.sqrt(2)
        nb_blocks = 3

        channels = [
            (
                int(input_size * (factor**i)),
                int(input_size * (factor ** (i + 1))),
            )
            for i in range(nb_blocks)
        ]

        super().__init__(
            neuron_number,
            channels[-1][1],
            unfolding_steps,
            activation_function,
            output_size,
        )

        self.__causal_encoder = nn.Sequential(
            *[CausalConvBlock(c_i, c_o) for c_i, c_o in channels]
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__causal_encoder(i.transpose(1, 2)).transpose(
            1, 2
        )
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]
