# -*- coding: utf-8 -*-
from math import sqrt
from statistics import mean
from typing import List

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from .causal import CausalConv1d
from .cell import LiquidCell
from .norm import TimeNorm


class LiquidRecurrent(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.__cell = LiquidCell(neuron_number, input_size, unfolding_steps)
        self.__neuron_number = neuron_number
        self.__to_output = nn.Linear(neuron_number, output_size)

    def __get_first_x(self, batch_size: int) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        return F.mish(
            th.randn(batch_size, self.__neuron_number, device=device)
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return i

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        out = self.__to_output(out)
        return out

    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, -1)

    def forward(self, i: th.Tensor, delta_t: th.Tensor) -> th.Tensor:
        b, _, _ = i.size()

        x_t = self.__get_first_x(b)
        i = self._process_input(i)

        results = []

        for t in range(i.size(2)):
            x_t = self.__cell(x_t, i[:, :, t], delta_t[:, t])
            results.append(self._output_processing(x_t))

        return self._sequence_processing(results)

    def count_parameters(self) -> int:
        return sum(
            int(np.prod(p.size()))
            for p in self.parameters()
            if p.requires_grad
        )

    def grad_norm(self) -> float:
        return mean(
            float(p.grad.norm().item())
            for p in self.parameters()
            if p.grad is not None
        )


class LiquidRecurrentReg(LiquidRecurrent):
    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return th.sigmoid(super()._output_processing(out))


class LiquidRecurrentBrainActivity(LiquidRecurrent):
    def __init__(
        self,
        neuron_number: int,
        input_size: int,
        unfolding_steps: int,
        output_size: int,
    ) -> None:
        nb_layer = 6
        factor = sqrt(2)
        encoder_dim = 16

        channels = [
            (
                int(encoder_dim * factor**i),
                int(encoder_dim * factor ** (i + 1)),
            )
            for i in range(nb_layer)
        ]

        super().__init__(
            neuron_number,
            channels[-1][1],
            unfolding_steps,
            output_size,
        )

        self.__conv = nn.Sequential(
            CausalConv1d(input_size, channels[0][0], dilation=1),
            nn.Mish(),
            TimeNorm(channels[0][0]),
            *[
                nn.Sequential(
                    nn.AvgPool1d(2, 2),
                    CausalConv1d(c_i, c_o, dilation=1),
                    nn.Mish(),
                    TimeNorm(c_o),
                )
                for c_i, c_o in channels
            ]
        )

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv(i)
        return out

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return F.softmax(super()._output_processing(out), dim=-1)

    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]


class LiquidRecurrentLast(LiquidRecurrent):
    def _sequence_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]
