from typing import Callable

import torch as th
from torch import nn
from torch.nn import functional as th_f

from ..abstract_recurent import AbstractLiquidRecurrent
from ..factory import AbstractLiquidRecurrentFactory
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
        nb_layers: int,
        factor: float,
        dropout: float,
    ) -> None:
        channels = [
            (int(input_size * factor**i), int(input_size * factor ** (i + 1)))
            for i in range(nb_layers)
        ]

        output_size = 6

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

        self.__dropout = nn.Dropout(dropout)
        self.__cls_token = nn.Parameter(th.randn(1, 1, channels[-1][1]))

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        encoded_input = self.__conv_encoder(i.transpose(1, 2)).transpose(1, 2)
        return th.cat(
            [
                encoded_input,
                self.__cls_token.repeat(encoded_input.size(0), 1, 1),
            ],
            dim=1,
        )

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th_f.log_softmax(
            super()._output_processing(self.__dropout(out)), dim=-1
        )

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class BrainActivityLiquidRecurrentFactory(
    AbstractLiquidRecurrentFactory[th.Tensor]
):
    def get_recurrent(
        self,
        neuron_number: int,
        unfolding_steps: int,
        act_fn: Callable[[th.Tensor], th.Tensor],
    ) -> AbstractLiquidRecurrent[th.Tensor]:
        return BrainActivityLiquidRecurrent(
            neuron_number,
            self._get_config("input_size", int),
            unfolding_steps,
            act_fn,
            self._get_config("nb_layers", int),
            self._get_config("factor", float),
            self._get_config("dropout", float),
        )
