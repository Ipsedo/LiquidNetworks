# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from statistics import mean
from typing import List

import numpy as np
import torch as th
from torch import nn

from .cell import LiquidCell


class LiquidRecurrent(ABC, nn.Module):
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
        return th.tanh(
            th.randn(batch_size, self.__neuron_number, device=device)
        )

    @abstractmethod
    def _output_activation(self, out: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def _outputs_post_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        pass

    def forward(self, i: th.Tensor, delta_t: th.Tensor) -> th.Tensor:
        b, _, steps = i.size()

        x_t = self.__get_first_x(b)

        results = []

        for t in range(steps):
            x_t = self.__cell(x_t, i[:, :, t], delta_t[:, t])
            results.append(self._output_activation(self.__to_output(x_t)))

        return self._outputs_post_processing(results)

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
    def _output_activation(self, out: th.Tensor) -> th.Tensor:
        return th.sigmoid(out)

    def _outputs_post_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, -1)


class LiquidRecurrentClf(LiquidRecurrent):
    def _output_activation(self, out: th.Tensor) -> th.Tensor:
        return out  # cross entropy perform softmax before nll

    def _outputs_post_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return th.stack(outputs, -1)


class LiquidRecurrentSingleClf(LiquidRecurrent):
    def _output_activation(self, out: th.Tensor) -> th.Tensor:
        return out  # cross entropy perform softmax before nll

    def _outputs_post_processing(self, outputs: List[th.Tensor]) -> th.Tensor:
        return outputs[-1]
