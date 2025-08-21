import math
from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as th_f

from .simple import LiquidRecurrent


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(
            th_f.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0))
        )


class CausalConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            CausalConv1d(in_channels, out_channels, 3, 1),
            nn.Mish(),
            CausalConv1d(out_channels, out_channels, 3, 2),
            nn.Mish(),
        )


class BrainActivityLiquidRecurrent(LiquidRecurrent):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        activation_function: Callable[[th.Tensor], th.Tensor],
        output_size: int,
    ) -> None:
        nb_layers = 4
        factor = math.sqrt(2.0)

        channels = [
            (int(input_size * factor**i), int(input_size * factor ** (i + 1)))
            for i in range(nb_layers)
        ]

        super().__init__(
            neuron_number,
            channels[-1][1],
            unfolding_steps,
            activation_function,
            output_size,
        )

        self.__conv_encoder = nn.Sequential(
            *[CausalConvBlock(c_i, c_o) for c_i, c_o in channels]
        )

        self.__pooling = nn.Linear(neuron_number, 1)

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        encoded_input: th.Tensor = self.__conv_encoder(
            i.transpose(1, 2)
        ).transpose(1, 2)
        return encoded_input

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        stacked_outputs = th.stack(outputs, dim=1)
        return th_f.softmax(
            super()._output_processing(
                th.sum(
                    stacked_outputs
                    * th_f.softmax(self.__pooling(stacked_outputs), dim=1),
                    dim=1,
                )
            ),
            dim=-1,
        )
